# ScrapJan

## NLP Web Scraper (WIP)

This is an experimental NLP web scraper designed to scrape specified keywords from web pages. It includes options for specifying HTML tags and provides outputs in a structured format.

---

## Requirements

Ensure you have the necessary packages installed. You can install them via:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Script

To run the scraper, use:

```bash
python main.py
```

### 2. Interactive Input

When prompted, provide the following inputs:

- **URL**: The web page URL you wish to scrape.
- **Keywords**: Enter the keywords (comma-separated) you want to match content with.
- **HTML Elements**: Specify HTML elements to search (e.g., `p`, `h1`, `h2`). By default, the script searches `p`, `h1`, `h2`, `h3`, `h4`, and `span`.

### 3. Output

After scraping, results will be displayed in the console. Extracted content that matches your keywords within the specified HTML elements will be printed.

---

## Example

```bash
python main.py
```

**Input:**
- URL: `https://example.com`
- Keywords: `data, AI, machine learning`
- HTML Elements: `p, h2`

**Console Output:**
```plaintext
Scraping URL: https://example.com
Keywords: data, AI, machine learning
HTML Elements: p, h2

Results:
1. "Data-driven insights in AI are transforming industries..." - [Found in `<p>` tag]
2. "Machine learning advancements in recent years..." - [Found in `<h2>` tag]
```

---

## Notes

- **Keyword Sensitivity**: The script is case-insensitive when searching for keywords.
- **HTML Parsing**: Make sure the URL you are scraping is accessible and has a valid HTML structure.
- **Performance**: Large pages may take longer to process, depending on the depth and number of elements searched.

---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to check the `issues` tab or submit a pull request.

---

## Author

- xStFtx

---