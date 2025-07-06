import os
import requests # نگهداری برای موارد احتمالی دیگر یا اگر کاربر بخواهد دوباره به آن برگردد
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from huggingface_hub import InferenceClient # وارد کردن InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# توکن‌های خود را اینجا قرار دهید
TELEGRAM_TOKEN = "-"
HF_TOKEN = "-" # توکن Hugging Face شما

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
    await update.message.reply_text(f"سلام {user_name}! 👋\nمن به هوش مصنوعی Hugging Face متصل هستم. سوالت رو بپرس.")

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
    processing_message = await update.message.reply_text("🧠 در حال پردازش سوال شما... (بار اول ممکن است کمی طول بکشد)")

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
            response_text = "⚠️ پاسخ غیرمنتظره‌ای از سرویس دریافت شد."

    except HfHubHTTPError as e:
        # مدیریت خطاهای HTTP خاص از Hugging Face Hub
        print(f"Hugging Face API HTTP Error: {e}")
        error_details = str(e)

        if "is currently loading" in error_details:
            response_text = "⏳ مدل در حال بارگذاری است. لطفاً حدود یک دقیقه دیگر دوباره تلاش کنید."
        else:
            response_text = f"⚠️ متاسفانه در ارتباط با سرویس هوش مصنوعی خطایی رخ داد: {error_details}"
    except Exception as e:
        # مدیریت خطاهای کلی
        print(f"General Error: {e}")
        response_text = "⚠️ متاسفانه یک خطای کلی در ربات رخ داد."

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

    print("Bot is running with updated error handling using Hugging Face InferenceClient...")
    # شروع نظرسنجی برای به‌روزرسانی‌ها
    application.run_polling()

if __name__ == '__main__':
    main()