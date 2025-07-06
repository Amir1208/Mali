import os
import requests # نگهداری برای موارد احتمالی دیگر یا اگر کاربر بخواهد دوباره به آن برگردد
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from huggingface_hub import InferenceClient # وارد کردن InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# توکن‌های خود را اینجا قرار دهید
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") # توکن Hugging Face شما
# WEBHOOK_URL = os.getenv("WEBHOOK_URL")
RANDOM_SECRET_TOKEN = os.getenv("RANDOM_SECRET_TOKEN")

# متغیرهای مربوط به Webhook
# اینها را هم به عنوان متغیر محیطی در Render تنظیم کنید
PORT = int(os.environ.get("PORT", 10000)) # Render یک متغیر PORT رو در اختیارتون میذاره، معمولا 10000
WEBHOOK_PATH = "/webhook" # مسیری که تلگرام به اون درخواست میفرسته

# این WEBHOOK_URL رو بعد از استقرار سرویس در Render و گرفتن آدرس عمومی، در تنظیمات Render ست کنید
# مثال: https://your-render-service-name.onrender.com/webhook
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", f"http://localhost:{PORT}{WEBHOOK_PATH}")

MODEL_ID = "google/gemma-2b-it"

# --- مقداردهی اولیه InferenceClient ---
# از HF_TOKEN مستقیماً استفاده می‌کنیم، همانطور که در کد اصلی شما تعریف شده است.
# اگر می‌خواهید از متغیر محیطی استفاده کنید، os.environ["HF_TOKEN"] را جایگزین کنید.
client = InferenceClient(
    provider="together", # این خط را می‌توان حذف کرد مگر اینکه نیاز خاصی به آن باشد
    api_key=HF_TOKEN # استفاده از توکن Hugging Face

)

# تابع برای دستور /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """این تابع پیام خوش‌آمدگویی را هنگام شروع ربات ارسال می‌کند."""
    user_name = update.message.from_user.first_name
    await update.message.reply_text(f"سلام {user_name}! 👋 ملیجکم.")

BOT_USERNAME = "@MaliJakEdarBar_bot"
# --- تابع اصلاح‌شده برای پردازش پیام ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """این تابع پیام‌های کاربر را پردازش کرده و پاسخ مدل هوش مصنوعی را برمی‌گرداند."""
    text = update.message.text or ""
    chat_type = update.message.chat.type

    # در گروه و سوپرگروه فقط وقتی نام بات منشن شده باشد پاسخ می‌دهد
    if chat_type in ("group", "supergroup"):
        if BOT_USERNAME not in text:
            return  # نادیده گرفتن پیام‌های بدون منشن
        # حذف منشن از متن برای ارسال به مدل
        user_message = text.replace(BOT_USERNAME, "").strip()
    else:
        # در چت خصوصی، تمام پیام‌ها پاسخ داده می‌شوند
        user_message = text

    # نمایش پیام "در حال پردازش..."
    processing_message = await update.message.reply_text("🧠 در حال پردازش سوال شما...")

    try:
        # استفاده از InferenceClient برای فراخوانی مدل
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            # می‌توانید پارامترهای دیگری مانند max_new_tokens, temperature و غیره را اینجا اضافه کنید
            # max_new_tokens=250,
            # temperature=0.7,
        )

        # استخراج متن تولید شده از پاسخ
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            response_text = completion.choices[0].message.content
        else:
            # اگر ساختار پاسخ غیرمنتظره بود
            print(f"Unexpected API Response Structure: {completion}")
            response_text = "⚠️ پاسخ غیرمنتظره"

    except HfHubHTTPError as e:
        # مدیریت خطاهای HTTP خاص از Hugging Face Hub
        print(f"Hugging Face API HTTP Error: {e}")
        error_details = str(e)

        if "is currently loading" in error_details:
            response_text = "⏳ در حال بارگذاری است..."
        else:
            response_text = f"⚠️ خطا: {error_details}"
    except Exception as e:
        # مدیریت خطاهای کلی
        print(f"General Error: {e}")
        response_text = "⚠️ یک خطای کلی رخ داد"

    # ویرایش پیام "در حال پردازش..." با پاسخ نهایی
    await context.bot.edit_message_text(chat_id=update.message.chat_id,
                                        message_id=processing_message.message_id,
                                        text=response_text)

def main() -> None:
    """نقطه ورود اصلی برای اجرای ربات تلگرام."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # اضافه کردن هندلرها
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"Bot is starting in webhook mode on port {PORT}...")
    # شروع نظرسنجی برای به‌روزرسانی‌ها
    # application.run_polling()

    # شروع ربات با Webhook
    application.run_webhook(
        listen="0.0.0.0",           # گوش دادن به تمام اینترفیس‌ها
        port=PORT,                  # پورت اختصاص داده شده توسط Render
        url_path=WEBHOOK_PATH,      # مسیر داخلی برای webhook
        webhook_url=WEBHOOK_URL,    # URL عمومی که تلگرام درخواست‌ها رو به اون میفرسته
        secret_token=RANDOM_SECRET_TOKEN # توصیه میشه از یک توکن تصادفی برای امنیت استفاده کنید
                                                # و اون رو هم به عنوان متغیر محیطی ست کنید.
    )

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return 'Bot is running!'

if __name__ == '__main__':
    main()
    # port = int(os.environ.get('PORT', 10000)) # Default to 10000 if PORT isn't set
    # app.run(host='0.0.0.0', port=port)
