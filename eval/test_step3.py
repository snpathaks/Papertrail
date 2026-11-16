import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from unittest.mock import patch, MagicMock
from agents.gap_agent import find_contradictions, find_candidates, PRIORITY
from tools.pdf_parser import download_pdf, get_chunks
import config

class TestStep3(unittest.TestCase):
    def test_find_contradictions(self):
        papers = [
            {
                "ref": "P1",
                "datasets": ["ETTm1"],
                "metrics": [{"name": "MSE", "value": "0.38", "dataset": "ETTm1"},
                            {"name": "Accuracy", "value": "91.2%", "dataset": "ETTm1"}]
            },
            {
                "ref": "P2",
                "datasets": ["ETTm1"],
                "metrics": [{"name": "Accuracy", "value": "0.75", "dataset": "ETTm1"}] # 75.0% vs 91.2% -> diff 16.2 >= 10
            }
        ]
        contradictions = find_contradictions(papers, min_diff=10)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0]["type"], "contradiction")
        self.assertIn("ettm1", contradictions[0]["finding"].lower())
        self.assertIn("75.0", contradictions[0]["finding"])
        self.assertIn("91.2", contradictions[0]["finding"])
        self.assertEqual(contradictions[0]["papers"], ["P2", "P1"])

    def test_priority_contradiction(self):
        self.assertIn("contradiction", PRIORITY)
        self.assertEqual(PRIORITY["contradiction"], 0)

    def test_pdf_download_size_cap(self):
        # Test file size exceeding 15MB cap
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": str(20 * 1024 * 1024)} # 20MB
            mock_get.return_value.__enter__.return_value = mock_resp
            
            path, err = download_pdf("http://example.com/oversized.pdf", "test_oversized")
            self.assertIsNone(path)
            self.assertIn("15 MB cap", err)

    def test_pdf_download_timeout(self):
        import requests
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
            path, err = download_pdf("http://example.com/slow.pdf", "test_timeout")
            self.assertIsNone(path)
            self.assertIn("timed out", err.lower())

    def test_get_chunks_fallback_issue(self):
        # Paper without PDF URL should return abstract chunk and fallback issue
        paper = {"id": "test_1", "abstract": "Test abstract content."}
        chunks, issue = get_chunks(paper, "test_1")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "abstract")
        self.assertIsNotNone(issue)
        self.assertTrue(issue["fallback"])
        self.assertIn("No PDF URL", issue["reason"])

    def test_extract_papers_progress_and_issues(self):
        from agents.extraction_agent import extract_papers
        progress_calls = []
        def mock_cb(msg):
            progress_calls.append(msg)
        
        state = {
            "selected": [
                {"id": "p1", "title": "Paper One", "authors": ["Author"], "year": 2023, "abstract": "Abstract 1"},
                {"id": "p2", "title": "Paper Two", "authors": ["Author"], "year": 2024, "abstract": "Abstract 2"}
            ],
            "progress_callback": mock_cb
        }
        with patch("agents.extraction_agent.extract_one") as mock_extract:
            mock_extract.side_effect = [
                {"id": "p1", "title": "Paper One", "pdf_fallback": True, "fallback_reason": "No PDF URL", "ref": "P1"},
                {"id": "p2", "title": "Paper Two", "pdf_fallback": False, "ref": "P2"}
            ]
            res = extract_papers(state)
            self.assertEqual(len(res["extracted"]), 2)
            self.assertEqual(len(progress_calls), 2)
            self.assertEqual(progress_calls[0], "Extracting 1/2")
            self.assertEqual(progress_calls[1], "Extracting 2/2")
            self.assertEqual(len(res["issues"]), 1)
            self.assertEqual(res["issues"][0]["title"], "Paper One")
            self.assertIn("No PDF URL", res["issues"][0]["reason"])

if __name__ == "__main__":
    unittest.main()
