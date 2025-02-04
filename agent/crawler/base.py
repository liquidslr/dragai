import asyncio
import os
import json
import hashlib
from collections import deque
from typing import List, Set

from crawl4ai import (
    AsyncWebCrawler,
)

from dragai.agent.crawler.config import Crawl4AIConfig
from dragai.agent.crawler.sitemeta import SitemapCrawler, RobotsHandler
from dragai.agent.crawler import URLUtils


class WebCrawler:
    def __init__(self, crawler_config=None, politness=2, max_concurrent=10):
        self.url_utils = URLUtils()
        self.sitemap_crawler = SitemapCrawler()
        self.robots_handler = RobotsHandler()
        self.politeness = politness
        self.max_concurrent = max_concurrent
        self.crawler_config = Crawl4AIConfig()

        if crawler_config:
            self.crawler_config = crawler_config

    async def get_page_urls(
        self,
        url: str,
        filename,
        same_origin=True,
    ) -> Set[str]:

        async with AsyncWebCrawler(
            config=self.crawler_config.browser_config
        ) as crawler:
            result = await crawler.arun(
                url=url, config=self.crawler_config.crawl_config
            )

            if result.success:
                filename = f"""./data/{filename}.jsonl"""

                if len(result.links.get("internal")):
                    internal_links = result.links.get("internal")

                    filtered_links = [
                        link["href"]
                        for link in internal_links
                        if self.crawler_config.filter_links(link.get("href", ""))
                    ]

                    with open(filename, "a", encoding="utf-8") as f:
                        for link in filtered_links:
                            json_line = json.dumps(link)
                            f.write(f"{json_line}\n")
                else:
                    link = self.crawler_config.filter_links(
                        result.markdown_v2.raw_markdown
                    )
                    with open(filename, "a") as f:
                        f.write(json.dumps(link))

    async def get_website_urls(
        self, url: str, same_origin=True, max_depth=5, max_urls=1000
    ) -> Set[str]:
        """Get all the links from the origin URL."""

        site_urls = set()
        seen_urls = set()
        orig_domain = self.url_utils.extract_domain(url)
        queue = deque([(url, 0)])

        async with AsyncWebCrawler(
            config=self.crawler_config.browser_config
        ) as crawler:
            while queue and len(site_urls) < max_urls:
                curr_url, depth = queue.popleft()
                if depth > max_depth or curr_url in seen_urls:
                    continue
                seen_urls.add(curr_url)

                print(f"Crawling ({depth}): {curr_url}")
                try:
                    result = await crawler.arun(
                        url, config=self.crawler_config.crawl_config
                    )
                    if result.success:
                        for link in result.links.get("internal", []):
                            normalized = self.url_utils.normalize_url(link["href"])
                            if self.url_utils.is_valid_url(normalized):
                                site_urls.add(normalized)
                                if (
                                    not same_origin
                                    or self.url_utils.extract_domain(normalized)
                                    == orig_domain
                                ):
                                    queue.append((normalized, depth + 1))
                    else:
                        print(f"[ERROR] {result.error_message}")
                except Exception as e:
                    print(f"[EXCEPTION] Error crawling {curr_url}: {e}")
        return site_urls

    async def crawl_parallel(
        self,
        urls: List[str],
        folder: str = None,
        filename: str = None,
        append: bool = False,
    ):
        """Crawl multiple URLs in parallel with a concurrency limit."""

        async with AsyncWebCrawler(
            config=self.crawler_config.browser_config
        ) as crawler:

            semaphore = asyncio.Semaphore(self.politeness)

            async def process_url(url: str):
                async with semaphore:
                    await asyncio.sleep(self.politeness)
                    try:
                        result = await crawler.arun(
                            url=url,
                            config=self.crawler_config.crawl_config,
                            session_id="session1",
                        )

                        if result.success:
                            print(f"Successfully crawled: {url}")

                            if append:
                                if filename is None:
                                    raise Exception("Filename is required")
                                file_name = filename
                            else:
                                file_name = (
                                    hashlib.sha256(url.encode()).hexdigest()[:5]
                                    if filename is None
                                    else filename
                                )

                            if folder is not None:
                                file_name = os.path.join(folder, file_name)

                            with open(
                                f"""{file_name}.md""", "a" if append else "w"
                            ) as f:
                                if self.crawler_config.filter_text:
                                    f.write(
                                        self.crawler_config.filter_text(
                                            result.markdown_v2.raw_markdown
                                        )
                                    )
                                else:
                                    f.write(result.markdown_v2.fit_markdown)
                        else:
                            print(f"Failed: {url} - Error: {result.error_message}")
                    except Exception as e:
                        print(f"Exception while crawling {url}: {e}")

            await asyncio.gather(*[process_url(url) for url in urls])

    async def get_data(self, url, filename):
        all_urls = self.sitemap_crawler.crawl_sitemap(url)
        if not len(all_urls):
            all_urls = await self.get_website_urls(url, filename)
            await self.crawl_parallel(list(all_urls), filename)

    async def get_page_data(
        self,
        url: str,
        folder: str = None,
        filename: str = None,
        append: bool = False,
    ):
        await self.crawl_parallel([url], folder, filename, append)
