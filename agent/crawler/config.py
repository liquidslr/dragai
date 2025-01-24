from crawl4ai import (
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    DefaultMarkdownGenerator,
)
from crawl4ai.content_filter_strategy import PruningContentFilter


class Crawl4AIConfig:
    def __init__(
        self,
        filter_text=None,
        filter_links=None,
        md_generator=None,
        browser_config=None,
        crawl_config=None,
        content_filter=None,
    ):

        if md_generator is None:
            self.md_generator = DefaultMarkdownGenerator(
                options={"ignore_links": False, "protect_links": False},
                content_filter=(
                    PruningContentFilter(threshold=0.6)
                    if content_filter is None
                    else content_filter
                ),
            )
        else:
            self.md_generator = md_generator

        if browser_config is None:
            self.browser_config = BrowserConfig(
                verbose=True,
                headless=False,
            )
        else:
            self.browser_config = browser_config

        if crawl_config is None:
            self.crawl_config = CrawlerRunConfig(
                markdown_generator=self.md_generator,
                cache_mode=CacheMode.BYPASS,
            )
        else:
            self.crawl_config = crawl_config

        self.filter_text = filter_text
        self.filter_links = filter_links
