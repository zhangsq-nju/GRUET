import json
import re
import urllib.error
import urllib.request

from ghapi.all import GhApi

from sweagent.utils.log import get_logger

_logger = get_logger("swea-github", emoji="🔧")

_repo_privacy_cache: dict[str, bool] = {}

GITHUB_ISSUE_URL_PATTERN = re.compile(r"github\.com\/(.*?)\/(.*?)\/issues\/(\d+)")


class InvalidGithubURL(Exception):
    pass


GITHUB_REPO_URL_PATTERN = re.compile(r".*[/@]?github\.com\/([^/]+)\/([^/]+)")


def _is_github_repo_url(data_path: str) -> bool:

    return GITHUB_REPO_URL_PATTERN.search(data_path) is not None


def _is_github_issue_url(data_path: str) -> bool:

    return GITHUB_ISSUE_URL_PATTERN.search(data_path) is not None


def _get_commit(api: GhApi, owner: str, repo: str, ref: str | None = None):

    if ref:
        return api.repos.get_commit(owner, repo, ref)
    return api.repos.list_commits(owner, repo)[0]


def _parse_gh_issue_url(issue_url: str) -> tuple[str, str, str]:

    match = GITHUB_ISSUE_URL_PATTERN.search(issue_url)
    if not match:
        msg = f"Invalid GitHub issue URL: {issue_url}"
        raise InvalidGithubURL(msg)
    res = match.groups()
    assert len(res) == 3
    return tuple(res)


def _parse_gh_repo_url(repo_url: str) -> tuple[str, str]:

    match = GITHUB_REPO_URL_PATTERN.search(repo_url)
    if not match:
        msg = f"Invalid GitHub issue URL: {repo_url}"
        raise InvalidGithubURL(msg)
    res = match.groups()
    assert len(res) == 2
    return tuple(res)


def _get_gh_issue_data(issue_url: str, *, token: str = ""):

    owner, repo, issue_number = _parse_gh_issue_url(issue_url)
    api = GhApi(token=token)
    return api.issues.get(owner, repo, issue_number)


def _get_problem_statement_from_github_issue(
    owner: str, repo: str, issue_number: str, *, token: str | None = ""
) -> str:

    api = GhApi(token=token)
    issue = api.issues.get(owner, repo, issue_number)
    title = issue.title if issue.title else ""
    body = issue.body if issue.body else ""
    return f"{title}\n{body}\n"


def _get_associated_commit_urls(org: str, repo: str, issue_number: str, *, token: str = "") -> list[str]:

    api = GhApi(token=token)


    events = api.issues.list_events(org, repo, issue_number)
    commit_urls = []
    for event in events:
        if event.event != "referenced":
            continue
        if not event.commit_id:
            continue
        commit = api.repos.get_commit(org, repo, event.commit_id)
        message = commit.commit.message
        if f"fixes #{issue_number}" in message.lower() or f"closes #{issue_number}" in message.lower():
            commit_urls.append(commit.html_url)
    return commit_urls


def _is_repo_private(owner_repo: str, token: str) -> bool:

    if owner_repo in _repo_privacy_cache:
        return _repo_privacy_cache[owner_repo]
    url = f"https://api.github.com/repos/{owner_repo}"
    headers = {"User-Agent": "sweagent"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            private = data.get("private", False)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            _logger.warning("Repo '%s' returned 404 — assuming private", owner_repo)
            private = True
        else:
            raise
    _repo_privacy_cache[owner_repo] = private
    return private
