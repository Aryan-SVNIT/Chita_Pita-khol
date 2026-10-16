import os
import re
import mimetypes
import tempfile
import requests
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

from flask import Flask, request
from dotenv import load_dotenv

from fact_checker import fact_check
from pdf_input import extract_pdf_text
from image_input import extract_image_text
from audio_input import transcribe_audio
from url_input import extract_article
from text_normalizer import normalize_extracted_text

load_dotenv()
app = Flask(__name__)

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v23.0")

MAX_MEDIA_BYTES = 15 * 1024 * 1024
URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)


@app.get("/webhook")
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge or "", 200
    return "Verification failed", 403


def _headers():
    if not ACCESS_TOKEN:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN is not configured")
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": str(message)[:4000]},
    }
    try:
        response = requests.post(
            url, headers={**_headers(), "Content-Type": "application/json"},
            json=payload, timeout=20,
        )
        if response.ok:
            print("WhatsApp reply sent.")
        else:
            print("WhatsApp API error:", response.status_code, response.text[:1000])
    except Exception as error:
        print("WhatsApp send failed:", repr(error))


def _download_whatsapp_media(media_id, declared_mime=""):
    """Resolve media ID via Graph API, then download it with bearer auth."""
    meta_url = f"https://graph.facebook.com/{API_VERSION}/{media_id}"
    meta_response = requests.get(meta_url, headers=_headers(), timeout=20)
    meta_response.raise_for_status()
    media_url = meta_response.json().get("url")
    mime = meta_response.json().get("mime_type") or declared_mime
    if not media_url:
        raise ValueError("Meta did not return a media download URL.")

    with requests.get(media_url, headers=_headers(), timeout=45, stream=True) as response:
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_MEDIA_BYTES:
            raise ValueError("File is too large. Please send a file under 15 MB.")
        chunks, total = [], 0
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_MEDIA_BYTES:
                raise ValueError("File is too large. Please send a file under 15 MB.")
            chunks.append(chunk)
    return b"".join(chunks), mime


def _extension_for(mime, filename=""):
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    mime = (mime or "").split(";")[0].strip().lower()
    known = {
        "application/pdf": ".pdf",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
    }
    return known.get(mime) or mimetypes.guess_extension(mime) or ".bin"


def _extract_message_text(message):
    """Return (extracted_text, metadata) for a supported WhatsApp message."""
    kind = message.get("type")

    if kind == "text":
        body = (message.get("text") or {}).get("body", "").strip()
        urls = URL_RE.findall(body)
        if urls and len(body) < 2500:
            url = urls[0].rstrip(".,!?;:")
            title, article_text, error = extract_article(url)
            if not error and article_text:
                return article_text, {"input_type": "url", "source": url, "title": title}
            # If URL extraction fails, fact-check the text/URL as supplied.
        return body, {"input_type": "text", "source": None, "title": None}

    if kind not in {"image", "document", "audio", "video"}:
        raise ValueError(
            "Supported inputs: text, webpage URL, image, PDF document, audio or voice note."
        )

    media_info = message.get(kind) or {}
    media_id = media_info.get("id")
    if not media_id:
        raise ValueError("WhatsApp did not provide a media ID.")
    declared_mime = media_info.get("mime_type", "")
    filename = media_info.get("filename", "")

    content, mime = _download_whatsapp_media(media_id, declared_mime)
    extension = _extension_for(mime, filename)

    with tempfile.TemporaryDirectory(prefix="chita_pita_") as temp_dir:
        path = Path(temp_dir) / ("incoming" + extension)
        path.write_bytes(content)

        if kind == "image" or (mime or "").startswith("image/"):
            text = extract_image_text(str(path))
            input_type = "image"
        elif extension == ".pdf" or mime == "application/pdf":
            text = extract_pdf_text(str(path))
            input_type = "pdf"
        elif kind == "audio" or (mime or "").startswith("audio/") or kind == "video":
            # Whisper handles audio and many voice-note formats; video containers
            # may also work if FFmpeg is installed.
            text = transcribe_audio(str(path), language_choice="3")
            input_type = "audio"
        else:
            raise ValueError("Unsupported file type. Send an image, PDF, or audio/voice note.")

        if not text or not str(text).strip():
            raise ValueError(
                f"I received your {input_type} file, but couldn't extract readable text. "
                "Try a clearer image, a text-based PDF, or a clearer audio recording."
            )
        return str(text), {
            "input_type": input_type,
            "source": f"WhatsApp attachment ({kind})",
            "title": filename or None,
        }


def format_fact_check_report(item, index, input_type="text"):
    claim = str(item.get("claim") or "Unknown")
    verdict = str(item.get("verdict") or "Unknown")
    confidence = str(item.get("confidence") or "Unknown")
    explanation = str(item.get("explanation") or "No explanation was provided.")

    reply = (
        "━━━━━━━━━━━━━━━━━━\n"
        "🔎 *CHITA-PITA KHOL*\n"
        f"🧾 *FACT-CHECK REPORT {index}*\n"
        f"📥 *INPUT: {input_type.upper()}*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        f"📝 *CLAIM*\n{claim[:650]}\n\n"
        f"⚖️ *VERDICT: {verdict.upper()}*\n"
        f"📊 *CONFIDENCE: {confidence.title()}*\n\n"
        f"💡 *EXPLANATION*\n{explanation[:650]}"
    )
    sources = item.get("sources") or []
    if sources:
        reply += "\n\n🔗 *SOURCES*"
        for number, source in enumerate(sources[:3], start=1):
            if not isinstance(source, dict):
                continue
            title = str(source.get("title") or "Source")
            url = str(source.get("url") or "").strip()
            reply += f"\n\n{number}. {title[:120]}"
            if url:
                reply += f"\n{url[:400]}"
    reply += (
        "\n\n━━━━━━━━━━━━━━━━━━\n"
        "_AI-assisted assessment. Please verify sources independently._"
    )
    return reply


def process_fact_check(sender, message):
    try:
        text, metadata = _extract_message_text(message)
        text = normalize_extracted_text(text or "")
        if not text:
            send_whatsapp_message(sender, "I couldn't extract readable text. Please try another file.")
            return

        send_whatsapp_message(
            sender,
            f"🔎 *Chita-Pita Khol*\nReceived your {metadata.get('input_type', 'text')} input. "
            "I'm extracting claims and checking evidence. Please wait..."
        )

        report = fact_check(text, metadata)
        claims = report.get("claims", [])
        if not claims:
            send_whatsapp_message(
                sender,
                "I couldn't identify a checkable factual claim in that input. "
                "Please send a clear claim or a file containing factual statements."
            )
            return

        for index, item in enumerate(claims[:10], start=1):
            send_whatsapp_message(
                sender,
                format_fact_check_report(item, index, metadata.get("input_type", "text")),
            )
        if len(claims) > 5:
            send_whatsapp_message(sender, "ℹ️ Only the first 10 claims are shown in WhatsApp.")
    except Exception as error:
        print("Fact-check/media processing error:", repr(error))
        send_whatsapp_message(
            sender,
            "Sorry, I couldn't process that input.\n"
            "Supported: text, webpage URL, image, PDF, audio or voice note.\n"
            "Check the server logs if the issue continues."
        )


@app.post("/webhook")
def receive_message():
    data = request.get_json(silent=True) or {}
    if data.get("object") != "whatsapp_business_account":
        return "EVENT_RECEIVED", 200

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "messages":
                continue
            value = change.get("value", {})
            for message in value.get("messages", []):
                sender = message.get("from")
                if not sender:
                    continue
                # Ignore delivery/read status events; process only actual messages.
                Thread(
                    target=process_fact_check,
                    args=(sender, message),
                    daemon=True,
                ).start()
    return "EVENT_RECEIVED", 200


@app.get("/")
def health_check():
    return "Chita-Pita Khol webhook is running!", 200


if __name__ == "__main__":
    for key, value in {
        "WHATSAPP_VERIFY_TOKEN": VERIFY_TOKEN,
        "WHATSAPP_ACCESS_TOKEN": ACCESS_TOKEN,
        "WHATSAPP_PHONE_NUMBER_ID": PHONE_NUMBER_ID,
    }.items():
        if not value:
            raise RuntimeError(f"Set {key} in your environment/.env")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
