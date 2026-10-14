from database.models import (
    ANALYSIS_MODEL_VERSIONS,
)

from database.repositories import (
    delete_analysis as repo_delete_analysis,
    get_analysis as repo_get_analysis,
    get_video_history as repo_get_video_history,
    list_analyses as repo_list_analyses,
    save_analysis as repo_save_analysis,
)


def save_analysis_history(
    result,
):
    """
    Prefer model versions reported by the live ABSA runtime.

    This is important because ABSA_RUNTIME_VERSION can be switched
    between v1 and v2 for rollback without changing database code.
    """
    return repo_save_analysis(
        result,

        ate_model_version=(
            result.get(
                "ate_model_version"
            )
            or ANALYSIS_MODEL_VERSIONS[
                "ate"
            ]
        ),

        asc_model_version=(
            result.get(
                "asc_model_version"
            )
            or ANALYSIS_MODEL_VERSIONS[
                "asc"
            ]
        ),
    )


def list_history(
    limit=20,
    offset=0,
):
    return repo_list_analyses(
        limit=limit,
        offset=offset,
    )


def get_history_detail(
    analysis_id,
):
    return repo_get_analysis(
        analysis_id
    )


def get_history_by_video(
    video_id,
    limit=20,
):
    return repo_get_video_history(
        video_id,
        limit=limit,
    )


def delete_history(
    analysis_id,
):
    return repo_delete_analysis(
        analysis_id
    )
