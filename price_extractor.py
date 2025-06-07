import asyncio
from playwright.async_api import async_playwright
import csv

product_urls = [
    "https://www.flipkart.com/tecno-pova-curve-5g-geek-black-128-gb/p/itma403c7d655267",
    "https://www.flipkart.com/lava-agni-2-glass-viridian-256-gb/p/itm1a372522c134e",
]


async def extract_flipkart_data(page, url):
    await page.goto(url, timeout=60000)
    #   await page.wait_for_selector('div[class*="price"]', timeout=20000)
    # Try common selectors for price and images
    price = None
    # Try multiple selectors for price
    selectors = [
        "div.Nx9bqj.CxhGGd",  # Selects the price div directly
        "div.hl05eU > div.Nx9bqj.CxhGGd",  # Selects the price div as a direct child
        'div[class*="hl05eU"] div.Nx9bqj.CxhGGd',  # More flexible, in case of nesting
    ]
    for sel in selectors:
        try:
            price_elem = await page.query_selector(sel)
            if price_elem:
                price = await price_elem.inner_text()
                break
        except:
            continue

    name_selectors = [
        "span.VU-ZEz",  # Most specific: the span with the name
        "h1._6EBuvT > span",  # Direct child span of h1 with class
        "h1._6EBuvT span",  # Any descendant span of h1 with class
    ]

    # For name
    name = None
    for sel in name_selectors:
        try:
            name_elem = await page.query_selector(sel)
            if name_elem:
                name = await name_elem.inner_text()
                break
        except Exception:
            continue

    highlights_selectors = [
        "div.xFVion li._7eSDEz",  # Most specific: li inside div.xFVion
        "ul > li._7eSDEz",  # li with class as direct child of ul
        "li._7eSDEz",  # Any li with that class
    ]
    # For highlights
    highlights = []
    for sel in highlights_selectors:
        try:
            highlight_elems = await page.query_selector_all(sel)
            if highlight_elems:
                highlights = [await elem.inner_text() for elem in highlight_elems]
                break
        except Exception:
            continue

    # Combine highlights as a single string (for CSV)
    highlights_str = " | ".join([h.strip() for h in highlights])

    # Get images under .MfqIAz
    images = []
    try:
        await page.wait_for_selector(".MfqIAz img", timeout=5000)
        img_elems = await page.query_selector_all(".MfqIAz img")
        for img in img_elems:
            src = await img.get_attribute("src") or await img.get_attribute("data-src")
            if src:
                images.append(src)
    except:
        pass

    return {
        "url": url,
        "price": price,
        "images": ";".join(images),
        "name": name,
        "highlights": highlights_str,
    }


async def main():
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for url in product_urls:
            data = await extract_flipkart_data(page, url)
            results.append(data)
        await browser.close()

    # Save to CSV
    with open("flipkart_products.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["url", "price", "images", "name", "highlights"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Scraping complete. Results saved to flipkart_products.csv.")


if __name__ == "__main__":
    asyncio.run(main())
