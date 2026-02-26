from src.main import build_parser


def test_parser_accepts_offline_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--topic", "t", "--docs", "a.txt", "--offline"])
    assert args.offline is True
    assert args.topic == "t"
    assert args.docs == ["a.txt"]
