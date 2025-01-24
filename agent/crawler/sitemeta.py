import requests
from collections import deque
from typing import List, Set
from xml.etree import ElementTree

from dragai.agent.crawler import URLUtils


class SitemapCrawler:
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """Get URLs from sitemap."""
        try:
            response = requests.get(sitemap_url)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            return [loc.text for loc in root.findall(".//ns:loc", namespace)]
        except Exception as e:
            print(f"Error fetching sitemap: {e}")
            return []

    def crawl_sitemap(self, url: str) -> Set[str]:
        """Crawl the sitemap for all links, handling nested .xml sitemaps."""
        visited = set()
        all_urls = set()
        queue = deque([URLUtils.extract_domain(url) + "/sitemap.xml"])

        while queue:
            current_sitemap = queue.pop()
            if current_sitemap in visited:
                continue
            visited.add(current_sitemap)
            links = self.parse_sitemap(current_sitemap)
            for link in links:
                if link.endswith(".xml"):
                    queue.append(link)
                else:
                    all_urls.add(link)
        return all_urls


class RobotsHandler:
    def parse_robotstxt(self, robots_url: str) -> List[str]:
        """Get disallowed URLs from robots.txt."""
        domain = URLUtils.extract_domain(robots_url)
        disallowed_urls = []
        try:
            response = requests.get(robots_url, timeout=10)
            response.raise_for_status()
            for line in response.text.splitlines():
                line = line.split("#", 1)[0].strip()
                if line.lower().startswith("disallow:"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        path = parts[1].strip()
                        if path:
                            disallowed_urls.append(f"{domain}{path}")
        except requests.RequestException as e:
            print(f"Error fetching robots.txt: {e}")
        return disallowed_urls

    def crawl_robotstxt(self, url: str) -> Set[str]:
        """Crawl the robots.txt for disallowed links."""
        domain = URLUtils.extract_domain(url)
        robots_url = f"{domain}/robots.txt"
        return set(self.parse_robotstxt(robots_url))
