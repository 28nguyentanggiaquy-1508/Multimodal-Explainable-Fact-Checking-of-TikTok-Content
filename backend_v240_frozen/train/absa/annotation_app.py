import json
import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


DATA_DIR = (
    BACKEND_DIR
    / "data"
    / "absa"
    / "annotated"
)

SOURCE_FILE = (
    DATA_DIR
    / "annotation_candidates.jsonl"
)

OUTPUT_FILE = (
    DATA_DIR
    / "annotated.jsonl"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ABSA Annotation Tool V2"
)


# ============================================================
# SCHEMAS
# ============================================================

class AspectInput(BaseModel):
    term: str
    start: int
    end: int
    polarity: str


class AnnotationInput(BaseModel):
    comment_id: str
    has_aspect: bool
    aspects: List[AspectInput] = []
    annotation_status: str = "done"


# ============================================================
# FILE UTILITIES
# ============================================================

def read_jsonl(path: Path):
    rows = []

    if not path.exists():
        return rows

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return rows


def load_candidates():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {SOURCE_FILE}"
        )

    return read_jsonl(
        SOURCE_FILE
    )


def load_annotations():

    rows = read_jsonl(
        OUTPUT_FILE
    )

    return {
        str(row.get("comment_id")): row
        for row in rows
    }


def save_annotations(annotation_map):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in annotation_map.values():

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# DATA MERGE
# ============================================================

def merged_rows():

    candidates = load_candidates()

    annotations = load_annotations()

    result = []

    for index, candidate in enumerate(
        candidates
    ):

        row = dict(
            candidate
        )

        comment_id = str(
            candidate.get(
                "comment_id"
            )
        )

        existing = annotations.get(
            comment_id
        )

        if existing:
            row.update(
                existing
            )

        row["index"] = index

        # Nếu chưa annotate
        if comment_id not in annotations:

            row["annotation_status"] = (
                "pending"
            )

            row["has_aspect"] = None

            row["aspects"] = []

        result.append(
            row
        )

    return result


# ============================================================
# FILTER
# ============================================================

def filter_rows(
    rows,
    filter_name: str,
):

    filter_name = (
        filter_name
        or "all"
    ).lower()

    if filter_name == "all":
        return rows

    if filter_name == "pending":

        return [
            r
            for r in rows
            if r.get(
                "annotation_status"
            ) == "pending"
        ]

    if filter_name == "done":

        return [
            r
            for r in rows
            if r.get(
                "annotation_status"
            ) == "done"
        ]

    if filter_name == "review":

        return [
            r
            for r in rows
            if r.get("annotation_status")
            in {"review", "needs_review"}
        ]

    if filter_name == "has_aspect":

        return [
            r
            for r in rows
            if r.get(
                "has_aspect"
            ) is True
        ]

    if filter_name == "no_aspect":

        return [
            r
            for r in rows
            if r.get(
                "has_aspect"
            ) is False
        ]

    return rows


# ============================================================
# API - STATS
# ============================================================

@app.get("/api/progress")
def progress():

    rows = merged_rows()

    total = len(
        rows
    )

    pending = sum(
        r.get("annotation_status")
        == "pending"
        for r in rows
    )

    done = sum(
        r.get("annotation_status")
        == "done"
        for r in rows
    )

    review = sum(
        r.get("annotation_status")
        == "review"
        for r in rows
    )

    has_aspect = sum(
        r.get("has_aspect")
        is True
        for r in rows
    )

    no_aspect = sum(
        r.get("has_aspect")
        is False
        for r in rows
    )

    return {
        "total": total,
        "pending": pending,
        "done": done,
        "review": review,
        "has_aspect": has_aspect,
        "no_aspect": no_aspect,
    }


# ============================================================
# API - FILTERED LIST
# ============================================================

@app.get("/api/list")
def list_rows(
    filter: str = Query(
        "all"
    )
):

    rows = merged_rows()

    filtered = filter_rows(
        rows,
        filter
    )

    return {
        "filter": filter,
        "total": len(filtered),

        "items": [
            {
                "index": row["index"],

                "comment_id":
                    row.get(
                        "comment_id"
                    ),

                "text":
                    row.get(
                        "text"
                    ),

                "annotation_status":
                    row.get(
                        "annotation_status"
                    ),

                "has_aspect":
                    row.get(
                        "has_aspect"
                    ),
            }

            for row in filtered
        ],
    }


# ============================================================
# API - SINGLE COMMENT
# ============================================================

@app.get("/api/comment/{index}")
def get_comment(
    index: int
):

    rows = merged_rows()

    if (
        index < 0
        or index >= len(rows)
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Comment not found"
            ),
        )

    row = dict(
        rows[index]
    )

    row["total"] = len(
        rows
    )

    return row


# ============================================================
# API - SAVE
# ============================================================

@app.post("/api/save")
def save_annotation(
    payload: AnnotationInput
):

    candidates = load_candidates()

    candidate = next(
        (
            row
            for row in candidates
            if str(
                row.get(
                    "comment_id"
                )
            )
            == str(
                payload.comment_id
            )
        ),
        None,
    )

    if candidate is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Comment not found"
            ),
        )

    status = (
        payload.annotation_status
        .lower()
        .strip()
    )

    if status not in {
        "done",
        "review",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid annotation status"
            ),
        )

    text = str(
        candidate.get(
            "text"
        )
        or ""
    )

    aspects = []

    for aspect in payload.aspects:

        polarity = (
            aspect.polarity
            .upper()
            .strip()
        )

        if polarity not in {
            "POSITIVE",
            "NEGATIVE",
            "NEUTRAL",
        }:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid polarity"
                ),
            )

        if (
            aspect.start < 0
            or aspect.end
            > len(text)
            or aspect.start
            >= aspect.end
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid aspect span"
                ),
            )

        actual_term = text[
            aspect.start:
            aspect.end
        ]

        aspects.append({
            "term":
                actual_term,

            "start":
                aspect.start,

            "end":
                aspect.end,

            "polarity":
                polarity,
        })

    # nếu nói có aspect
    # nhưng không có aspect nào
    if (
        payload.has_aspect
        and not aspects
        and status == "done"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Has aspect=True nhưng "
                "không có aspect"
            ),
        )

    row = {

        "comment_id":
            candidate.get(
                "comment_id"
            ),

        "video_id":
            candidate.get(
                "video_id"
            ),

        "text":
            text,

        "has_aspect":
            payload.has_aspect,

        "aspects":
            aspects
            if payload.has_aspect
            else [],

        "annotation_status":
            status,
    }

    annotations = (
        load_annotations()
    )

    annotations[
        str(
            payload.comment_id
        )
    ] = row

    save_annotations(
        annotations
    )

    return {
        "success": True,
        "data": row,
    }


# ============================================================
# UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def annotation_ui():

    return HTMLResponse(
        content=HTML_PAGE
    )


HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="vi">

<head>

<meta charset="UTF-8">

<title>
ABSA Annotation Tool V2
</title>

<style>

body {
    font-family:
        Arial,
        sans-serif;

    background:
        #f1f5f9;

    margin: 0;
}

.container {
    max-width: 1050px;
    margin: 25px auto;
}

.card {
    background: white;
    padding: 25px;

    border-radius: 14px;

    box-shadow:
        0 3px 14px
        rgba(0,0,0,0.08);
}

h2 {
    margin-top: 0;
}

.stats {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 18px;
}

.stat {
    padding:
        8px 12px;

    background:
        #f8fafc;

    border-radius:
        8px;

    border:
        1px solid #e2e8f0;
}

.filters {
    margin-bottom:
        20px;
}

button {
    padding:
        9px 14px;

    margin:
        4px;

    border:
        none;

    border-radius:
        7px;

    cursor:
        pointer;
}

button:hover {
    opacity: 0.85;
}

.filter-button {
    background:
        #e2e8f0;
}

.filter-button.active {
    background:
        #2563eb;

    color:
        white;
}

.comment-meta {
    color:
        #64748b;

    margin-bottom:
        8px;
}

.comment {
    font-size:
        24px;

    line-height:
        1.6;

    padding:
        24px;

    border-radius:
        10px;

    background:
        #f8fafc;

    border:
        1px solid #e2e8f0;

    user-select:
        text;
}

.selection-box {
    margin-top:
        18px;

    padding:
        16px;

    border:
        1px solid #ddd;

    border-radius:
        8px;
}

.positive {
    background:
        #dcfce7;
}

.neutral {
    background:
        #e5e7eb;
}

.negative {
    background:
        #fee2e2;
}

.primary {
    background:
        #2563eb;

    color:
        white;
}

.warning {
    background:
        #fbbf24;
}

.no-aspect {
    background:
        #64748b;

    color:
        white;
}

.aspect-item {
    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        10px;

    margin-top:
        8px;

    background:
        #f1f5f9;

    border-radius:
        7px;
}

.aspect-positive {
    border-left:
        5px solid #16a34a;
}

.aspect-neutral {
    border-left:
        5px solid #64748b;
}

.aspect-negative {
    border-left:
        5px solid #dc2626;
}

.navigation {
    margin-top:
        20px;
}

.status-line {
    margin-top:
        10px;

    font-weight:
        bold;
}

.review-text {
    color:
        #b45309;
}

.done-text {
    color:
        #15803d;
}

.pending-text {
    color:
        #64748b;
}

</style>

</head>


<body>

<div class="container">

<div class="card">

<h2>
ABSA Annotation Tool V2
</h2>


<div
    id="stats"
    class="stats">
</div>


<div class="filters">

<button
    class="filter-button"
    onclick="setFilter('all', this)">
All
</button>

<button
    class="filter-button"
    onclick="setFilter('pending', this)">
Pending
</button>

<button
    class="filter-button"
    onclick="setFilter('done', this)">
Done
</button>

<button
    class="filter-button"
    onclick="setFilter('review', this)">
Review
</button>

<button
    class="filter-button"
    onclick="setFilter('has_aspect', this)">
Has Aspect
</button>

<button
    class="filter-button"
    onclick="setFilter('no_aspect', this)">
No Aspect
</button>

</div>


<div
    id="commentMeta"
    class="comment-meta">
</div>


<div
    id="comment"
    class="comment">
</div>


<div
    id="annotationStatus"
    class="status-line">
</div>


<div class="selection-box">

<div>
Selected aspect:

<strong
    id="selectedText">
Chưa chọn
</strong>

</div>


<div>

<button
    class="positive"
    onclick="
        addAspect('POSITIVE')
    ">
POSITIVE
</button>

<button
    class="neutral"
    onclick="
        addAspect('NEUTRAL')
    ">
NEUTRAL
</button>

<button
    class="negative"
    onclick="
        addAspect('NEGATIVE')
    ">
NEGATIVE
</button>

</div>

</div>


<h3>
Aspects
</h3>

<div id="aspects">
</div>


<div class="navigation">

<button
    class="primary"
    onclick="saveDone()">
Save / Update
</button>

<button
    class="no-aspect"
    onclick="saveNoAspect()">
No Aspect
</button>

<button
    class="warning"
    onclick="saveReview()">
Needs Review
</button>

<br>

<button
    onclick="previousFiltered()">
← Previous
</button>

<button
    onclick="nextFiltered()">
Next →
</button>

</div>

</div>

</div>


<script>

let current = null;

let aspects = [];

let selection = null;

let currentFilter =
    "all";

let filteredItems = [];

let filteredPosition = 0;


// ============================================================
// STATS
// ============================================================

async function refreshStats() {

    const response =
        await fetch(
            "/api/progress"
        );

    const data =
        await response.json();

    document.getElementById(
        "stats"
    ).innerHTML = `

        <div class="stat">
        Total:
        <strong>
        ${data.total}
        </strong>
        </div>

        <div class="stat">
        Pending:
        <strong>
        ${data.pending}
        </strong>
        </div>

        <div class="stat">
        Done:
        <strong>
        ${data.done}
        </strong>
        </div>

        <div class="stat">
        Review:
        <strong>
        ${data.review}
        </strong>
        </div>

        <div class="stat">
        Has Aspect:
        <strong>
        ${data.has_aspect}
        </strong>
        </div>

        <div class="stat">
        No Aspect:
        <strong>
        ${data.no_aspect}
        </strong>
        </div>
    `;
}


// ============================================================
// FILTER
// ============================================================

async function setFilter(
    filterName,
    button
) {

    currentFilter =
        filterName;

    document
        .querySelectorAll(
            ".filter-button"
        )
        .forEach(
            b =>
                b.classList.remove(
                    "active"
                )
        );

    if (button) {
        button.classList.add(
            "active"
        );
    }

    const response =
        await fetch(
            `/api/list?filter=${filterName}`
        );

    const data =
        await response.json();

    filteredItems =
        data.items || [];

    filteredPosition = 0;

    if (
        filteredItems.length === 0
    ) {

        current = null;

        document.getElementById(
            "comment"
        ).innerText =
            "Không có comment trong filter này.";

        return;
    }

    await loadFilteredPosition(
        0
    );
}


// ============================================================
// LOAD
// ============================================================

async function loadFilteredPosition(
    position
) {

    if (
        position < 0
        ||
        position >=
            filteredItems.length
    ) {
        return;
    }

    filteredPosition =
        position;

    const item =
        filteredItems[
            position
        ];

    const response =
        await fetch(
            `/api/comment/${item.index}`
        );

    const data =
        await response.json();

    loadData(
        data
    );
}


function loadData(data) {

    current =
        data;

    aspects =
        data.aspects || [];

    selection =
        null;

    document.getElementById(
        "comment"
    ).innerText =
        data.text;

    document.getElementById(
        "selectedText"
    ).innerText =
        "Chưa chọn";

    document.getElementById(
        "commentMeta"
    ).innerText =
        `Dataset index: ${
            data.index + 1
        } / ${data.total}
        | Filter:
        ${
            filteredPosition + 1
        } / ${
            filteredItems.length
        }`;

    const status =
        data.annotation_status
        || "pending";

    const statusElement =
        document.getElementById(
            "annotationStatus"
        );

    statusElement.innerText =
        `Status: ${
            status.toUpperCase()
        }`;

    statusElement.className =
        "status-line";

    if (
        status === "done"
    ) {

        statusElement.classList.add(
            "done-text"
        );

    } else if (
        status === "review"
    ) {

        statusElement.classList.add(
            "review-text"
        );

    } else {

        statusElement.classList.add(
            "pending-text"
        );
    }

    renderAspects();
}


// ============================================================
// TEXT SELECTION
// ============================================================

document.addEventListener(
    "mouseup",
    function() {

        if (!current)
            return;

        const element =
            document.getElementById(
                "comment"
            );

        const sel =
            window.getSelection();

        if (
            !sel
            ||
            sel.rangeCount === 0
        )
            return;

        const selected =
            sel.toString();

        if (
            !selected.trim()
        )
            return;

        if (
            !element.contains(
                sel.anchorNode
            )
        )
            return;

        const range =
            sel.getRangeAt(0);

        const preRange =
            range.cloneRange();

        preRange.selectNodeContents(
            element
        );

        preRange.setEnd(
            range.startContainer,
            range.startOffset
        );

        const start =
            preRange
            .toString()
            .length;

        const end =
            start
            + selected.length;

        selection = {

            term:
                selected,

            start:
                start,

            end:
                end
        };

        document.getElementById(
            "selectedText"
        ).innerText =
            selected;
    }
);


// ============================================================
// ASPECT
// ============================================================

function addAspect(
    polarity
) {

    if (!selection) {

        alert(
            "Hãy bôi đen aspect trước."
        );

        return;
    }

    aspects.push({

        term:
            selection.term,

        start:
            selection.start,

        end:
            selection.end,

        polarity:
            polarity
    });

    selection =
        null;

    document.getElementById(
        "selectedText"
    ).innerText =
        "Chưa chọn";

    renderAspects();
}


function removeAspect(
    index
) {

    aspects.splice(
        index,
        1
    );

    renderAspects();
}


function cyclePolarity(
    index
) {

    const order = [
        "POSITIVE",
        "NEUTRAL",
        "NEGATIVE"
    ];

    const currentPolarity =
        aspects[index]
        .polarity;

    const currentIndex =
        order.indexOf(
            currentPolarity
        );

    aspects[index]
        .polarity =
        order[
            (
                currentIndex + 1
            )
            %
            order.length
        ];

    renderAspects();
}


function renderAspects() {

    const container =
        document.getElementById(
            "aspects"
        );

    container.innerHTML =
        "";

    aspects.forEach(
        (
            aspect,
            index
        ) => {

            const div =
                document.createElement(
                    "div"
                );

            const polarity =
                aspect.polarity
                .toLowerCase();

            div.className =
                `aspect-item
                 aspect-${polarity}`;

            div.innerHTML = `

                <div>

                    <strong>
                    ${aspect.term}
                    </strong>

                    →

                    ${aspect.polarity}

                    <small>
                    [${aspect.start},
                    ${aspect.end}]
                    </small>

                </div>

                <div>

                    <button
                    onclick="
                    cyclePolarity(
                        ${index}
                    )
                    ">
                    Change
                    </button>

                    <button
                    onclick="
                    removeAspect(
                        ${index}
                    )
                    ">
                    X
                    </button>

                </div>
            `;

            container.appendChild(
                div
            );
        }
    );
}


// ============================================================
// SAVE
// ============================================================

async function save(
    hasAspect,
    status
) {

    if (!current)
        return false;

    const payload = {

        comment_id:
            String(
                current.comment_id
            ),

        has_aspect:
            hasAspect,

        aspects:
            hasAspect
                ? aspects
                : [],

        annotation_status:
            status
    };

    const response =
        await fetch(
            "/api/save",
            {

                method:
                    "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(
                        payload
                    )
            }
        );

    if (!response.ok) {

        const error =
            await response.json();

        alert(
            JSON.stringify(
                error
            )
        );

        return false;
    }

    await refreshStats();

    return true;
}


async function saveDone() {

    if (
        aspects.length === 0
    ) {

        alert(
            "Không có aspect. "
            + "Nếu thật sự không có target "
            + "hãy chọn No Aspect."
        );

        return;
    }

    const success =
        await save(
            true,
            "done"
        );

    if (success) {
        await reloadFilterAfterSave();
    }
}


async function saveNoAspect() {

    aspects = [];

    const success =
        await save(
            false,
            "done"
        );

    if (success) {
        await reloadFilterAfterSave();
    }
}


async function saveReview() {

    const success =
        await save(
            aspects.length > 0,
            "review"
        );

    if (success) {
        await reloadFilterAfterSave();
    }
}


async function reloadFilterAfterSave() {

    const previousPosition =
        filteredPosition;

    const response =
        await fetch(
            `/api/list?filter=${
                currentFilter
            }`
        );

    const data =
        await response.json();

    filteredItems =
        data.items || [];

    if (
        filteredItems.length === 0
    ) {

        document.getElementById(
            "comment"
        ).innerText =
            "Đã xử lý hết filter này.";

        current = null;

        return;
    }

    let newPosition =
        previousPosition;

    if (
        newPosition >=
        filteredItems.length
    ) {

        newPosition =
            filteredItems.length
            - 1;
    }

    await loadFilteredPosition(
        newPosition
    );
}


// ============================================================
// NAVIGATION
// ============================================================

function previousFiltered() {

    if (
        filteredPosition > 0
    ) {

        loadFilteredPosition(
            filteredPosition - 1
        );
    }
}


function nextFiltered() {

    if (
        filteredPosition
        <
        filteredItems.length - 1
    ) {

        loadFilteredPosition(
            filteredPosition + 1
        );
    }
}


// ============================================================
// START
// ============================================================

async function start() {

    await refreshStats();

    const allButton =
        document.querySelector(
            ".filter-button"
        );

    await setFilter(
        "all",
        allButton
    );
}

start();

</script>

</body>

</html>
"""