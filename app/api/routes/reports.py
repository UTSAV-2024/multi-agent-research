from fastapi import APIRouter, HTTPException

from app.repositories.report_repository import (
    get_report_by_id,
    delete_report
)

from app.utils.response_builder import (
    success_response,
    error_response
)

from app.utils.logger import logger


router = APIRouter()


# ==========================================
# GET REPORT BY ID
# ==========================================

@router.get("/reports/{report_id}")
async def get_report(report_id: str):

    logger.info(
        f"[API] Fetching report | id={report_id}"
    )

    report = await get_report_by_id(report_id)

    if not report:

        logger.warning(
            f"[API] Report not found | id={report_id}"
        )

        raise HTTPException(
            status_code=404,
            detail={
                "status": "failed",
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Report {report_id} not found"
                }
            }
        )

    logger.info(
        f"[API] Report retrieved | id={report_id}"
    )

    return success_response(
        data=report,
        message="Report retrieved"
    )


# ==========================================
# DELETE REPORT
# ==========================================

@router.delete("/reports/{report_id}")
async def remove_report(report_id: str):

    logger.info(
        f"[API] Deleting report | id={report_id}"
    )

    deleted = await delete_report(report_id)

    if not deleted:

        logger.warning(
            f"[API] Report not found for deletion | "
            f"id={report_id}"
        )

        raise HTTPException(
            status_code=404,
            detail={
                "status": "failed",
                "error": {
                    "code": "NOT_FOUND",
                    "message": f"Report {report_id} not found"
                }
            }
        )

    logger.info(
        f"[API] Report deleted | id={report_id}"
    )

    return success_response(
        message="Report deleted"
    )
