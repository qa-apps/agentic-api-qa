"""Human-readable report rendering with screenshot evidence for every UI case."""

from __future__ import annotations

import base64
import html
import mimetypes
import os
from pathlib import Path
import shutil
from typing import Any
import webbrowser

from agentic_api_qa.models import DesignScore, FinalReport, UICaseResult


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _embedded_image(raw_path: str | None, *, alt: str) -> str:
    """Return a self-contained image so the HTML can be opened or shared alone."""

    if not raw_path:
        return '<div class="missing">Screenshot capture failed</div>'
    path = Path(raw_path)
    if not path.is_file():
        return f'<div class="missing">Screenshot not found: {_e(path.name)}</div>'
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    data_uri = f"data:{mime};base64,{encoded}"
    return (
        f'<a class="screenshot-link" href="{data_uri}" target="_blank" rel="noopener" '
        f'title="Open {_e(path.name)} at full size">'
        f'<img loading="lazy" src="{data_uri}" alt="{_e(alt)}"></a>'
    )


def _video_attachment(raw_path: str | None, *, destination: Path, run_id: str) -> str:
    if not raw_path:
        return ""
    source = Path(raw_path)
    if not source.is_file():
        return f'<div class="missing">Failure video not found: {_e(source.name)}</div>'
    run_directory = Path(run_id).name or "unknown"
    target = destination.parent / "videos" / run_directory / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    relative_path = target.relative_to(destination.parent).as_posix()
    mime = mimetypes.guess_type(source.name)[0] or "video/webm"
    return (
        '<div class="video-evidence"><h3>Failure video</h3>'
        f'<video controls preload="metadata"><source src="{_e(relative_path)}" '
        f'type="{_e(mime)}">Open <a href="{_e(relative_path)}">{_e(source.name)}</a>'
        "</video></div>"
    )


def _ui_case_cards(
    results: list[UICaseResult], *, destination: Path, run_id: str
) -> str:
    cards: list[str] = []
    for item in results:
        image = _embedded_image(
            item.screenshot_path,
            alt=f"Screenshot for {item.case_id}",
        )
        video = _video_attachment(
            item.video_path,
            destination=destination,
            run_id=run_id,
        )
        status = "pass" if item.passed else "fail"
        console = "\n".join(item.console_errors) or "No console errors."
        snapshot = item.snapshot_excerpt or "No accessibility snapshot captured."
        cards.append(
            f"""
            <article class="case {status}">
              <header><span class="badge">{_e(item.severity.value)}</span>
                <h2>{_e(item.name)}</h2><strong>{'PASS' if item.passed else 'FAIL'}</strong>
              </header>
              <p><code>{_e(item.case_id)}</code> · {_e(item.category)} · {item.duration_ms:.0f} ms</p>
              <p>{_e(item.reason)}</p>
              {image}
              <p class="screenshot-hint">Click the screenshot to open it at full size.</p>
              {video}
              <details><summary>Tools actually used</summary><pre>{_e(', '.join(item.mcp_tools))}</pre></details>
              <details><summary>Console evidence</summary><pre>{_e(console)}</pre></details>
              <details><summary>Accessibility snapshot</summary><pre>{_e(snapshot)}</pre></details>
            </article>
            """
        )
    return "".join(cards)


def _design_section(design: DesignScore | None) -> str:
    if not design:
        return ""
    strengths = "".join(f"<li>{_e(item)}</li>" for item in design.strengths)
    issues = "".join(f"<li>{_e(item)}</li>" for item in design.issues)
    return f"""
    <section class="design"><h2>Design evaluation</h2>
      <p>{_e(design.summary)}</p>
      <div class="scores">
        <span>Hierarchy {design.visual_hierarchy}/5</span>
        <span>Readability {design.readability}/5</span>
        <span>Consistency {design.consistency}/5</span>
        <span>Responsive {design.responsive_layout}/5</span>
        <span>Accessibility {design.accessibility_cues}/5</span>
        <span>Interactions {design.interaction_clarity}/5</span>
      </div>
      <div class="design-lists"><div><h3>Strengths</h3><ul>{strengths}</ul></div>
      <div><h3>Issues</h3><ul>{issues}</ul></div></div>
    </section>
    """


def render_html_report(report: FinalReport, destination: Path) -> Path:
    """Write a self-contained HTML report with every screenshot embedded."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cards = _ui_case_cards(
        report.ui_smoke,
        destination=destination,
        run_id=report.run.run_id,
    )
    design_html = _design_section(report.design_evaluation)
    judge_cards = []
    findings = {item.finding_id: item for item in report.candidate_findings}
    for decision in report.judge_decisions:
        finding = findings.get(decision.finding_id)
        judge_cards.append(
            f"""
            <article class="case {'fail' if decision.verdict.value == 'CONFIRMED' else 'pass'}">
              <header><span class="badge">{_e(decision.verdict.value)}</span>
                <h2>{_e(finding.title if finding else decision.finding_id)}</h2>
                <strong>{decision.confidence:.0%}</strong>
              </header>
              <p><strong>Skeptical challenge:</strong> {_e(decision.skeptical_challenge)}</p>
              <p><strong>Decision:</strong> {_e(decision.reasoning_summary)}</p>
              <p><strong>Investigation:</strong> {_e(decision.investigation_reason)}</p>
              <p>Healer eligible: <code>{_e(decision.healer_eligible)}</code></p>
            </article>
            """
        )
    healer_html = ""
    if report.healer_result:
        healer = report.healer_result
        evidence_images: list[str] = []
        for label, paths in (
            ("Before", healer.before_screenshots),
            ("After", healer.after_screenshots),
        ):
            for raw_path in paths:
                path = Path(raw_path)
                if path.is_file():
                    evidence_images.append(
                        f'<figure><figcaption>{_e(label)}: {_e(path.name)}</figcaption>'
                        f'{_embedded_image(str(path), alt=f"{label} fix evidence")}</figure>'
                    )
        healer_html = f"""
        <section class="design"><h1>Healer and human-review handoff</h1>
          <p><strong>Status:</strong> {_e(healer.status.value)}</p>
          <p><strong>Investigation reason:</strong> {_e(healer.investigation_reason)}</p>
          <p><strong>Investigation result:</strong> {_e(healer.investigation_result)}</p>
          <p><strong>Proposed fix:</strong> {_e(healer.proposed_fix)}</p>
          <p><strong>Draft PR:</strong> {_e(healer.pr_url or 'not created')}</p>
          <p><strong>Human review:</strong> required; this workflow cannot merge.</p>
          <div class="grid">{''.join(evidence_images)}</div>
          <details><summary>Validation</summary><pre>{_e(chr(10).join(healer.validation_commands + healer.validation_output))}</pre></details>
        </section>
        """
    reviewer_html = ""
    if report.reviewer_decision:
        reviewer = report.reviewer_decision
        scan = report.reviewer_scan
        gate_text = "not available"
        findings = ""
        if scan:
            gate_text = (
                f"secrets={scan.secret_scan_passed}, static={scan.static_scan_passed}, "
                f"dependencies={scan.dependency_scan_passed}, deletions={scan.deletion_guard_passed}; "
                f"diff +{scan.additions}/-{scan.deletions}"
            )
            findings = "\n".join(scan.findings) or "No deterministic findings."
        reviewer_html = f"""
        <section class="design"><h1>Automated pull-request Reviewer</h1>
          <p><strong>Verdict:</strong> {_e(reviewer.verdict.value)} ({reviewer.confidence:.0%})</p>
          <p><strong>Reviewed commit:</strong> {_e(reviewer.approved_commit_sha or 'not approved')}</p>
          <p><strong>Security:</strong> {_e(reviewer.security_assessment)}</p>
          <p><strong>Deletion/scope:</strong> {_e(reviewer.deletion_assessment)}</p>
          <p><strong>Quality:</strong> {_e(reviewer.quality_assessment)}</p>
          <p><strong>Deterministic gates:</strong> {_e(gate_text)}</p>
          <details><summary>Findings and required changes</summary><pre>{_e(findings + chr(10) + chr(10) + chr(10).join(reviewer.required_changes))}</pre></details>
          <p>Automated approval is first-pass only. Human approval remains mandatory.</p>
        </section>
        """
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Agentic QA report {_e(report.run.run_id)}</title>
<style>
body{{font:15px/1.5 system-ui;margin:0;background:#0b1020;color:#e8ecf4}}
main{{max-width:1200px;margin:auto;padding:28px}} h1,h2{{margin:.2em 0}}
.summary,.design,.case{{background:#151d31;border:1px solid #2b3858;border-radius:14px;padding:18px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
.case{{margin:0}} .case.pass{{border-left:5px solid #2fc878}} .case.fail{{border-left:5px solid #ff6577}}
header{{display:flex;gap:10px;align-items:center}} header h2{{flex:1;font-size:18px}}
.badge,.scores span{{background:#263455;border-radius:99px;padding:4px 9px}}
.scores{{display:flex;flex-wrap:wrap;gap:8px}} .design-lists{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.screenshot-link{{display:block;background:#0f1424;border-radius:8px;overflow:hidden}}
img{{display:block;width:100%;height:360px;object-fit:contain;background:#0f1424;border-radius:8px}}
video{{display:block;width:100%;max-height:480px;background:#0f1424;border-radius:8px}}
.screenshot-hint{{margin:.45rem 0 0;color:#aeb9d2;font-size:13px}}
.missing{{padding:60px;text-align:center;background:#351c28;color:#ffb4bf;border-radius:8px}}
code,pre{{white-space:pre-wrap;word-break:break-word}} a{{color:#93b4ff}}
</style></head><body><main>
<section class="summary"><h1>Agentic QA report</h1>
<p><strong>{_e(report.run.overall_result.value)}</strong> · {_e(report.run.target)}</p>
<p>{_e(report.summary)}</p>
<p>UI screenshots: {report.metrics.ui_screenshots}/{len(report.ui_smoke)} · Failure videos: {report.metrics.ui_videos} · MCP calls: {report.metrics.ui_mcp_tool_calls}</p>
</section>
{design_html}
<section><h1>Independent Judge</h1><div class="grid">{''.join(judge_cards) or '<p>No candidate defects.</p>'}</div></section>
{healer_html}
{reviewer_html}
<section><h1>UI test evidence</h1><div class="grid">{cards}</div></section>
</main></body></html>"""
    destination.write_text(body, encoding="utf-8")
    return destination


def render_ui_execution_report(payload: dict[str, Any], destination: Path) -> Path:
    """Render a standalone Playwright MCP execution payload as one visual HTML file."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    results = [UICaseResult.model_validate(item) for item in payload.get("results", [])]
    raw_design = payload.get("design_evaluation")
    design = DesignScore.model_validate(raw_design) if raw_design else None
    passed = sum(item.passed for item in results)
    screenshots = sum(
        bool(item.screenshot_path and Path(item.screenshot_path).is_file()) for item in results
    )
    videos = sum(bool(item.video_path and Path(item.video_path).is_file()) for item in results)
    run_id = str(payload.get("run_id", "unknown"))
    cards = _ui_case_cards(results, destination=destination, run_id=run_id)
    plan = payload.get("plan_summary") or "No agent plan summary was recorded."
    errors = payload.get("pipeline_errors") or []
    errors_html = _e("\n".join(str(item) for item in errors) or "No pipeline errors.")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>UI Agent report {_e(run_id)}</title>
<style>
body{{font:15px/1.5 system-ui;margin:0;background:#0b1020;color:#e8ecf4}}
main{{max-width:1400px;margin:auto;padding:28px}} h1,h2{{margin:.2em 0}}
.summary,.design,.case{{background:#151d31;border:1px solid #2b3858;border-radius:14px;padding:18px;margin:16px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
.case{{margin:0}} .case.pass{{border-left:5px solid #2fc878}} .case.fail{{border-left:5px solid #ff6577}}
header{{display:flex;gap:10px;align-items:center}} header h2{{flex:1;font-size:18px}}
.badge,.scores span{{background:#263455;border-radius:99px;padding:4px 9px}}
.scores{{display:flex;flex-wrap:wrap;gap:8px}} .design-lists{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.screenshot-link{{display:block;background:#0f1424;border-radius:8px;overflow:hidden}}
img{{display:block;width:100%;height:360px;object-fit:contain;background:#0f1424;border-radius:8px}}
video{{display:block;width:100%;max-height:480px;background:#0f1424;border-radius:8px}}
.screenshot-hint{{margin:.45rem 0 0;color:#aeb9d2;font-size:13px}}
.missing{{padding:60px;text-align:center;background:#351c28;color:#ffb4bf;border-radius:8px}}
code,pre{{white-space:pre-wrap;word-break:break-word}} a{{color:#93b4ff}}
</style></head><body><main>
<section class="summary"><h1>UI Agent execution report</h1>
<p><strong>{passed}/{len(results)} passed</strong> · {screenshots}/{len(results)} screenshots embedded · {videos} failure videos · run {_e(run_id)}</p>
<p><strong>Agent plan:</strong> {_e(plan)}</p>
<details><summary>Pipeline errors</summary><pre>{errors_html}</pre></details>
</section>
{_design_section(design)}
<section><h1>Executed test cases with evidence</h1><div class="grid">{cards}</div></section>
</main></body></html>"""
    destination.write_text(body, encoding="utf-8")
    return destination


def report_directory() -> Path:
    """Return the project-local directory used for visual and machine reports."""

    return Path(os.getenv("QA_REPORT_DIR", "reports")).resolve()


def publish_latest_report(html_report: Path, json_report: Path | None = None) -> Path:
    """Publish a stable Playwright-style entry point for the most recent run."""

    html_report = html_report.resolve()
    latest = html_report.parent / "index.html"
    if html_report != latest:
        latest.write_bytes(html_report.read_bytes())
    if json_report is not None:
        json_report = json_report.resolve()
        latest_json = html_report.parent / "latest.json"
        if json_report != latest_json:
            latest_json.write_bytes(json_report.read_bytes())
    return latest


def publish_report_bundle(
    report: FinalReport,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write one archived run plus stable latest HTML/JSON entry points."""

    destination = (directory or report_directory()).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_report = destination / f"qa-report-{report.run.run_id}.json"
    json_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_report = render_html_report(report, json_report.with_suffix(".html"))
    latest_html = publish_latest_report(html_report, json_report)
    return {
        "json": json_report,
        "html": html_report,
        "latest_html": latest_html,
        "latest_json": destination / "latest.json",
    }


def show_latest_report() -> None:
    """Open the stable latest visual report in the default browser."""

    latest = report_directory() / "index.html"
    if not latest.is_file():
        raise SystemExit(f"No visual report yet: {latest}")
    webbrowser.open(latest.as_uri())
    print(f"Opened visual report: {latest}")
