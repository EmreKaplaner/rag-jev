"""Real Chromium interactions against the HTTP service and controlled upstreams."""

import json

from playwright.sync_api import expect, sync_playwright


def connect(page, base_url):
    page.goto(base_url)
    page.get_by_text("Service access", exact=True).click()
    page.get_by_label("Service bearer token").fill("integration-test-token")
    page.get_by_role("button", name="Connect", exact=True).click()
    expect(page.locator("#query")).to_have_value("How many days do I have to request a refund?")


def test_browser_score_compare_replay_export_and_import(network_service, tmp_path):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        connect(page, network_service)
        page.get_by_role("button", name="Score with Jev").click()
        expect(page.locator("#status")).to_contain_text("Live scoring complete")
        expect(page.locator(".passage.kept")).to_have_count(2)
        expect(page.locator(".passage.dropped")).to_have_count(1)
        page.get_by_role("button", name="Generate both answers").click()
        expect(page.locator("#status")).to_contain_text("Comparison complete")
        expect(page.locator(".answer")).to_have_count(2)
        expect(page.locator(".answer").last).to_contain_text("controlled-http-model")
        with page.expect_download() as download:
            page.get_by_role("button", name="Export replay").click()
        path = tmp_path / "run.json"
        download.value.save_as(path)
        payload = json.loads(path.read_text())
        assert payload["comparison"]["selected"]["citations"] == ["refund", "receipt"]
        assert "integration-test-token" not in path.read_text()
        page.locator("#threshold").fill("0.9")
        page.locator("#threshold").dispatch_event("change")
        expect(page.locator("#status")).to_contain_text("Policy applied")
        expect(page.locator(".passage.kept")).to_have_count(1)
        expect(page.locator(".answer")).to_have_count(0)
        page.locator("#import").set_input_files(path)
        expect(page.locator("#status")).to_contain_text("Imported replay")
        expect(page.locator(".passage.kept")).to_have_count(2)
        expect(page.locator(".answer")).to_have_count(2)
        expect(page.locator("#provenance")).to_contain_text("IMPORTED")
        page.locator("#query").fill("An edited question")
        expect(page.locator("#compare")).to_be_disabled()
        expect(page.locator("#export")).to_be_disabled()
        assert errors == []
        browser.close()


def test_browser_contextual_scoring_and_strategy_change(network_service):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        connect(page, network_service)
        page.get_by_role("button", name="Try research v2 settings").click()
        expect(page.locator("#scoring-strategy")).to_have_value("contextual")
        expect(page.locator("#mode")).to_have_value("filter_and_rerank")
        expect(page.locator("#threshold")).to_have_value("0.2")
        expect(page.locator("#top-n")).to_have_value("")
        page.get_by_role("button", name="Score with Jev").click()
        expect(page.locator("#status")).to_contain_text("Live scoring complete")
        expect(page.locator("#provenance")).to_contain_text("contextual")
        expect(page.locator(".passage.kept")).to_have_count(2)
        page.get_by_role("button", name="Generate both answers").click()
        expect(page.locator(".answer")).to_have_count(2)
        page.locator("#scoring-strategy").select_option("independent")
        expect(page.locator("#compare")).to_be_disabled()
        expect(page.locator(".answer")).to_have_count(0)
        page.goto(network_service + "/benchmarks")
        expect(page.get_by_role("heading", name="Less context. Check the outcome.")).to_be_visible()
        expect(page.locator("table tbody tr")).to_have_count(6)
        page.goto(network_service + "/assets/research-v2.html")
        expect(
            page.get_by_role("heading", name="Keep the links. Measure the tradeoff.")
        ).to_be_visible()
        expect(page.locator("table tbody tr")).to_have_count(6)
        browser.close()


def test_browser_empty_fixture_mobile_and_literal_source_text(network_service):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        connect(page, network_service)
        page.locator("#example").select_option("no-answer")
        page.get_by_role("button", name="Explore hand-authored fixture").click()
        expect(page.locator("#status")).to_contain_text("Hand-authored fixture loaded")
        expect(page.locator(".passage.kept")).to_have_count(0)
        expect(page.locator("#policy-note")).to_contain_text("No context selected")
        page.get_by_role("button", name="Generate both answers").click()
        expect(page.locator("#status")).to_contain_text("Comparison complete")
        expect(page.locator(".answer").last).to_contain_text("Application fallback")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        # XSS-like source text is rendered literally through textContent, never innerHTML.
        fixture = page.request.get(
            network_service + "/v1/fixtures/refund",
            headers={"Authorization": "Bearer integration-test-token"},
        ).json()
        fixture["record"]["request"]["documents"][0]["text"] = (
            '<img src=x onerror="window.pwned=true">'
        )
        page.route("**/v1/fixtures/refund", lambda route: route.fulfill(json=fixture))
        page.locator("#example").select_option("refund")
        page.get_by_role("button", name="Explore hand-authored fixture").click()
        expect(page.locator("#evidence")).to_contain_text('<img src=x onerror="window.pwned=true">')
        assert page.locator("#evidence img").count() == 0
        assert page.evaluate("window.pwned") is None
        browser.close()


def test_browser_recorded_regression_budget_and_review_export(network_service, tmp_path):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        connect(page, network_service)
        page.locator("#recording").select_option("regression")
        expect(page.locator("#provenance")).to_contain_text("RECORDED LIVE RUN")
        expect(page.locator(".answer")).to_have_count(2)
        expect(page.locator("#status")).to_contain_text("regression")
        page.locator("#token-budget").fill("1")
        page.locator("#token-budget").dispatch_event("change")
        expect(page.locator("#status")).to_contain_text("Policy applied")
        expect(page.locator(".passage.kept")).to_have_count(0)
        expect(page.locator(".answer")).to_have_count(0)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        packet = tmp_path / "packet.json"
        packet.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": "q",
                            "query": "Test question",
                            "reference_answers": ["yes"],
                            "full_candidate_evidence": [{"text": "Public test evidence"}],
                            "answers": {"A": {"text": "yes"}, "B": {"text": "no"}},
                        }
                    ]
                }
            )
        )
        page.goto(network_service + "/review")
        page.locator("#packet").set_input_files(packet)
        expect(page.locator("#review-status")).to_contain_text("2 answers loaded")
        page.get_by_label("Reviewer name or identifier").fill("test-only-reviewer")
        page.get_by_label("Semantically correct?", exact=True).first.select_option("yes")
        page.get_by_label("Question or reference ambiguous?", exact=True).first.select_option("no")
        with page.expect_download() as download:
            page.get_by_role("button", name="Export review labels").click()
        destination = tmp_path / "labels.csv"
        download.value.save_as(destination)
        from rag_jev.review import summarize_review

        result = summarize_review(str(packet), str(destination))
        assert result["status"] == "incomplete" and result["labeled_answers"] == 1
        browser.close()
