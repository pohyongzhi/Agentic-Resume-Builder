import unittest
from unittest.mock import patch

import config
from common.prompts import BASE_RULES, clean_model_output


class InferenceBoundaryTests(unittest.TestCase):
    def test_v4_1_is_the_default_writer_model(self):
        self.assertEqual(config.DEFAULT_MODEL, "agentic-resume-v4-1:q4_K_M")
        self.assertEqual(config.BASE_MODEL, "qwen3.5:4b-q4_K_M")

    def test_writer_model_can_be_overridden_without_editing_code(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": config.BASE_MODEL}):
            self.assertEqual(config.os.getenv("OLLAMA_MODEL", config.DEFAULT_MODEL), config.BASE_MODEL)

    def test_terminal_control_tokens_are_removed(self):
        self.assertEqual(
            clean_model_output("## Skills\nJava<|im_end|>\n"),
            "## Skills\nJava",
        )
        self.assertEqual(
            clean_model_output("resume<|im_end|><|endoftext|>"),
            "resume",
        )

    def test_embedded_text_is_not_silently_removed(self):
        value = "Evidence includes <|im_end|> in the middle of a sentence."
        self.assertEqual(clean_model_output(value), value)

    def test_prompt_preserves_titles_and_proficiency_limits(self):
        self.assertIn("Do not adopt the target job title", BASE_RULES)
        self.assertIn("percentages, proportions, proficiency levels", BASE_RULES)
        self.assertIn("Use 'Not provided' only", BASE_RULES)


if __name__ == "__main__":
    unittest.main()
