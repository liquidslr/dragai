from urllib.parse import urlparse, urlunparse


EXCLUDED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".pdf",
    ".docx",
    ".xlsx",
    ".zip",
    ".rar",
    ".exe",
    ".svg",
    ".css",
    ".js",
]


class URLUtils:
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from a URL."""
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize the URL by removing fragments and queries."""
        parsed = urlparse(url)
        normalized = parsed._replace(fragment="", query="")
        return urlunparse(normalized).rstrip("/")

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate the URL."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False
        path = parsed.path.lower()
        return not any(path.endswith(ext) for ext in EXCLUDED_EXTENSIONS)
