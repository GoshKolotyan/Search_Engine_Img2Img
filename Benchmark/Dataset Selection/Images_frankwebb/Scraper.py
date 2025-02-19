import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

URLs = {
    "Tub_URLs": [],
    "Toilet_URLs": [],
    "Mirror_URLs": [],
    "Cabinet_URLs": [],
    "Sink_URLs": [
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?hawkcustom=undefined&pg=2",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?hawkcustom=undefined%2cundefined&pg=3",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?hawkcustom=undefined%2cundefined&pg=4",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?mpp=96&pg=5",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?mpp=96&pg=6",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-drop-in-bathroom-sinks-c30027?mpp=96&pg=7",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-undermount-bathroom-sinks-c30032?mpp=96",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-vessel-and-above-counter-sinks-c30033",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-vessel-and-above-counter-sinks-c30033?mpp=96&pg=2",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-vessel-and-above-counter-sinks-c30033?mpp=96&pg=3",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-vessel-and-above-counter-sinks-c30033?mpp=96&pg=4",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-vessel-and-above-counter-sinks-c30033?mpp=96&pg=5",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-wall-mount-sinks-c30034",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-wall-mount-sinks-c30034?mpp=96&pg=2",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-wall-mount-sinks-c30034?mpp=96&pg=3",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-pedestal-sink-sets-c30029",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-pedestal-sink-sets-c30029?mpp=96&pg=2",
        "https://frankwebb.com/collections/category-bathroom-bathroom-sinks-pedestal-sink-sets-c30029?mpp=96&pg=3"
    ],
}

# Folder to save images
SAVE_FOLDER = "Images_frankwebb"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode
driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

# Iterate over each category and its associated URL list
for category, url_list in URLs.items():
    print(f"Processing {category}")

    # Create subfolder for this category
    category_folder = os.path.join(SAVE_FOLDER, category)
    os.makedirs(category_folder, exist_ok=True)

    # Go through each URL in the current category
    for URL in url_list:
        print("Loading URL:", URL)
        driver.get(URL)
        print("Sleeping 20 second")
        time.sleep(20)  # Allow JavaScript content to load

        # Scroll down to load lazy-loaded images
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(15)  # Wait for images to load
        print("Exiting 15 second")
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # 1) Change the selector to match the new website's image tags
        product_images = soup.select("img.itemImage.hawk-itemImage")

        print(f"Found {len(product_images)} product images in {category}.")

        # Extract and save images
        for index, img in enumerate(product_images):
            img_url = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-lazy")
                or img.get("srcset")
            )
            # Skip any SVG placeholders if present
            if img_url and not img_url.startswith("data:image/svg+xml"):
                img_url = urljoin(URL, img_url)  # Ensure full URL
                print(f"Downloading: {img_url}")
                try:
                    img_data = requests.get(img_url).content

                    # Construct image filename
                    # Note: If your URL doesn't have "=", this might produce an empty part.
                    img_name = f"image_{index}_page__{URL[-1]}_{img_url.split('.')[-2].split('/')[-1]}.jpg"
                    # Save to the category subfolder
                    img_path = os.path.join(category_folder, img_name)
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    print(f"Saved: {img_name} in {category}")
                except Exception as e:
                    print(f"Failed to save {img_url}: {e}")
            else:
                print(f"Skipped: {img_url}")

driver.quit()
print("Scraping completed.")
