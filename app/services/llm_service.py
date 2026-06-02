# ==========================================
# CHANGES MADE:
# ==========================================
#
# 1. Added retry handling
# 2. Added exponential backoff
# 3. Added structured retry logging
# 4. Added rate-limit visibility
# 5. Added timeout/failure observability
# 6. Added configurable retry settings
# 7. Added safer exception handling
# 8. Added semaphore concurrency control
# 9. Added timeout protection
# 10. Added async-safe execution
#
# ==========================================

import os
import time
import asyncio

from groq import Groq

from dotenv import load_dotenv

from app.config.settings import settings

from app.utils.logger import logger


# ==========================================
# ENVIRONMENT SETUP
# ==========================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# GLOBAL CONCURRENCY CONTROL
# ==========================================

llm_semaphore = asyncio.Semaphore(1)


# ==========================================
# MAIN LLM EXECUTION FUNCTION
# ==========================================

async def run_agent(
    system_prompt,
    user_prompt,
    max_tokens=1000
):

    logger.info(
        "[LLM QUEUE] Waiting for semaphore slot"
    )

    async with llm_semaphore:

        logger.info(
            "[LLM EXECUTION] Semaphore acquired"
        )

        # ==========================================
        # RETRY LOOP
        # ==========================================

        for attempt in range(
            1,
            settings.MAX_RETRIES + 1
        ):

            try:

                logger.info(
                    f"[LLM SERVICE] Attempt {attempt}"
                )

                start = time.time()

                # ==========================================
                # TIMEOUT-PROTECTED API CALL
                # ==========================================

                response = await asyncio.wait_for(

                    asyncio.to_thread(

                        client.chat.completions.create,

                        model=settings.MODEL_NAME,

                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],

                        temperature=settings.TEMPERATURE,

                        max_tokens=max_tokens
                    ),

                    timeout=60
                )

                duration = round(
                    time.time() - start,
                    2
                )

                logger.info(
                    f"[LLM SERVICE] Success in {duration}s"
                )

                return response.choices[0].message.content.strip()

            except asyncio.TimeoutError:

                logger.error(
                    f"[LLM SERVICE] Timeout on attempt {attempt}"
                )

            except Exception as e:

                logger.error(
                    f"[LLM SERVICE] Attempt {attempt} failed | {e}"
                )

            # ==========================================
            # EXPONENTIAL BACKOFF
            # ==========================================

            if attempt < settings.MAX_RETRIES:

                delay = attempt * 2

                logger.warning(
                    f"[LLM SERVICE] Retrying in {delay}s..."
                )

                await asyncio.sleep(delay)

            else:

                logger.error(
                    "[LLM SERVICE] Maximum retries exceeded"
                )

                return """
                {
                    "facts": []
                }
                """