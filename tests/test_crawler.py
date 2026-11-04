"""Pure-function tests for the competitor crawler (no network)."""

from scripts.crawl_competitors import (
    PROVIDERS, Page, join_broken_lines, page_service_name, prefer_vietnamese,
    strip_boilerplate, table_chunks, url_is_candidate,
)


def test_url_filter_keeps_product_pages_and_drops_news():
    vnpt = PROVIDERS["vnpt"]
    assert url_is_candidate("https://cloud.vnpt.vn/dich-vu/object-storage-s3", vnpt)
    assert not url_is_candidate("https://cloud.vnpt.vn/blog/abc", vnpt)
    assert not url_is_candidate("https://cloud.vnpt.vn/tai-lieu/Redis-544", vnpt)
    assert not url_is_candidate("https://evil.example/dich-vu/x", vnpt)
    biz = PROVIDERS["bizfly"]
    assert url_is_candidate("https://bizflycloud.vn/cloud-server/bang-gia", biz)
    assert not url_is_candidate("https://bizflycloud.vn/tin-tuc/khuyen-mai.htm", biz)
    assert not url_is_candidate("https://bizflycloud.vn/tong-dai-ai/giao-duc", biz)


def test_own_site_is_not_a_default_competitor():
    from scripts.crawl_competitors import DEFAULT_PROVIDERS
    assert PROVIDERS["cmc"].role == "own" and "cmc" not in DEFAULT_PROVIDERS


def test_prefer_vietnamese_drops_english_twin():
    urls = ["https://x.vn/en/product/a", "https://x.vn/product/a", "https://x.vn/en/product/b"]
    assert prefer_vietnamese(urls) == ["https://x.vn/product/a", "https://x.vn/en/product/b"]


def test_join_broken_lines():
    text = join_broken_lines(["Dòng máy chủ giá", "rẻ đáp ứng nhu cầu.", "Gói Micro", "116.820", "đ/tháng"])
    assert text.splitlines() == ["Dòng máy chủ giá rẻ đáp ứng nhu cầu.", "Gói Micro", "116.820 đ/tháng"]


def test_strip_boilerplate_removes_lines_shared_by_most_pages():
    pages = [Page("p", f"u{i}", "t", "now", f"Menu\nFooter\nUnique content {i}") for i in range(4)]
    assert strip_boilerplate(pages) == 2
    assert all(p.text.startswith("Unique content") for p in pages)


def test_table_chunks_repeat_the_header():
    table = [["Gói", "Giá"]] + [[f"G{i}", f"{i}00.000 đ"] for i in range(25)]
    chunks = table_chunks(table, "bizfly - Cloud Server", rows_per_chunk=12)
    assert len(chunks) == 3
    assert all(c.splitlines()[1] == "Gói | Giá" for c in chunks)
    assert "G24 | 2400.000 đ" in chunks[-1]


def test_page_service_name():
    page = Page("vnpt", "https://cloud.vnpt.vn/dich-vu/x", "VNPT Cloud Object Storage | Lưu trữ S3", "now", "")
    assert page_service_name(page) == "vnpt - VNPT Cloud Object Storage"


def test_join_broken_lines_collapses_whitespace():
    assert join_broken_lines(["FPT ArgoCD 0          đ/ tháng", "   ", "Gói\u00a0\u00a0B"]) == "FPT ArgoCD 0 đ/ tháng\nGói B"
