#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `abalone_dpacia_sklearn` package."""


import unittest
from click.testing import CliRunner

from abalone_dpacia_sklearn import abalone_dpacia_sklearn
from abalone_dpacia_sklearn import cli


class TestAbalone_dpacia_sklearn(unittest.TestCase):
    """Tests for `abalone_dpacia_sklearn` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'abalone_dpacia_sklearn.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
