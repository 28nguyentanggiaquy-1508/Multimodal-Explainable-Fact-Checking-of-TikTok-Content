from database.db import (
    DB_PATH,
    init_db,
)

from database.repositories import (
    list_analyses,
)


def main():
    init_db()

    print(
        "Database:",
        DB_PATH,
    )

    print(
        "History:",
        list_analyses(
            limit=5
        ),
    )


if __name__ == "__main__":
    main()
