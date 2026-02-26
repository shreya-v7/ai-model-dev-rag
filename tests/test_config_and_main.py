from pathlib import Path

from src import config


def test_bootstrap_runtime_dirs_creates_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    config.bootstrap_runtime_dirs()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "outputs").exists()


def test_main_executes_offline(monkeypatch, tmp_path, capsys) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("simple corpus evidence for deterministic test", encoding="utf-8")
    out = tmp_path / "out.json"

    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBED_PROVIDER", "hash")

    import src.main as main_mod

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--topic",
            "topic",
            "--docs",
            str(doc),
            "--output",
            str(out),
            "--offline",
        ],
    )
    main_mod.main()
    assert Path(out).exists()
    printed = capsys.readouterr().out
    assert "topic_summary" in printed
