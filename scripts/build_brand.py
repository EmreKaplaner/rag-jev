"""Build self-contained SVG brand assets from the original squirrel mark; no network."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "src/rag_jev/static/brand"


def main():
    mark = (BRAND / "squirrel.svg").read_text()
    shapes = mark[mark.index("  <g") : mark.rindex("</svg>")]
    (BRAND / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
        '<title>rag-jev</title><rect width="128" height="128" rx="26" fill="#F5F1E7"/>'
        f'<g fill="none" transform="translate(2 0) scale(.96)">{shapes}</g></svg>\n'
    )
    (ROOT / "integrations/dify/_assets/icon.svg").write_text((BRAND / "favicon.svg").read_text())
    (BRAND / "banner.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="420" '
        'viewBox="0 0 1280 420" role="img" aria-labelledby="title desc">'
        '<title id="title">rag-jev — Make room for useful evidence.</title>'
        '<desc id="desc">An open-source context selection layer for RAG, with a russet '
        "squirrel mascot on warm cream paper.</desc>"
        '<rect width="1280" height="420" rx="16" fill="#F5F1E7"/>'
        '<path d="M56 64H780M56 350H1224" stroke="#CDD4C9"/>'
        '<text x="56" y="46" fill="#566D64" font-family="Verdana,sans-serif" '
        'font-size="12" letter-spacing="3">OPEN SOURCE / CONTEXT SELECTION</text>'
        '<text x="50" y="171" fill="#253B36" font-family="Georgia,serif" '
        'font-size="112" letter-spacing="-6">rag<tspan fill="#B84F2D">-</tspan>jev</text>'
        '<text x="56" y="244" fill="#253B36" font-family="Georgia,serif" '
        'font-size="40">Make room for useful evidence.</text>'
        '<text x="58" y="291" fill="#566D64" font-family="Verdana,sans-serif" '
        'font-size="17">A small layer between retrieval and generation.</text>'
        '<circle cx="1050" cy="184" r="144" fill="#E6ECDF"/>'
        f'<g fill="none" transform="translate(933 55) scale(1.9)">{shapes}</g>'
        '<text x="56" y="382" fill="#566D64" font-family="Verdana,sans-serif" '
        'font-size="12" letter-spacing="2">FILTER / RERANK / INSPECT / MEASURE</text>'
        '<text x="1224" y="382" text-anchor="end" fill="#566D64" '
        'font-family="Verdana,sans-serif" font-size="12">Python · HTTP · TypeScript · MIT</text>'
        "</svg>\n"
    )


if __name__ == "__main__":
    main()
