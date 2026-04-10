"""Entry point for the Visual Query Builder desktop app."""

from __future__ import annotations

from src.ui.query_builder import QueryBuilderApp


def main() -> None:
    app = QueryBuilderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
