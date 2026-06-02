from datetime import datetime

from bson import ObjectId

from app.db.collections.reports_collection import (
    reports_collection
)

from app.utils.logger import logger


# ==========================================
# SAVE REPORT
# ==========================================

async def save_report(report_data: dict):

    report_data["created_at"] = datetime.utcnow()

    result = await reports_collection.insert_one(
        report_data
    )

    logger.info(
        f"[REPO] Report saved | "
        f"id={result.inserted_id}"
    )

    return str(result.inserted_id)


# ==========================================
# GET HISTORY (paginated)
# ==========================================

async def get_history(
    limit: int = 20,
    skip: int = 0
):

    cursor = (
        reports_collection.find()
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )

    reports = []

    async for report in cursor:

        report["_id"] = str(report["_id"])

        reports.append(report)

    return reports


# ==========================================
# GET REPORT BY ID
# ==========================================

async def get_report_by_id(report_id: str):

    try:

        result = await reports_collection.find_one(
            {"_id": ObjectId(report_id)}
        )

        if result:

            result["_id"] = str(result["_id"])

        return result

    except Exception as e:

        logger.error(
            f"[REPO] Failed to fetch report "
            f"{report_id} | {e}"
        )

        return None


# ==========================================
# DELETE REPORT
# ==========================================

async def delete_report(report_id: str):

    try:

        result = await reports_collection.delete_one(
            {"_id": ObjectId(report_id)}
        )

        deleted = result.deleted_count > 0

        if deleted:

            logger.info(
                f"[REPO] Report deleted | "
                f"id={report_id}"
            )

        return deleted

    except Exception as e:

        logger.error(
            f"[REPO] Failed to delete report "
            f"{report_id} | {e}"
        )

        return False