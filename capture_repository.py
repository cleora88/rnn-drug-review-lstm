from pathlib import Path

from playwright.sync_api import sync_playwright

R = Path(__file__).parent
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto(
        "https://github.com/cleora88/rnn-drug-review-lstm",
        wait_until="domcontentloaded",
        timeout=60000,
    )
    assert "rnn-drug-review-lstm" in page.title()
    page.screenshot(
        path=str(R / "screenshots/06_public_repository.png"), full_page=False
    )
    print(page.title())
    browser.close()
