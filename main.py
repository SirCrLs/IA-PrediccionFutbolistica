# main.py
import argparse

from Fase1_preparacion import ejecutar_fase1
from Fase2_entrenamiento import ejecutar_fase2
from security import load_pkl, log, require, validate_match


def parse_args():
    parser = argparse.ArgumentParser(description="Secure football prediction pipeline")
    parser.add_argument("action", nargs="?", choices=("phase1", "phase2", "all", "predict"), default="all")
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--league", choices=("E0", "E1", "E2", "E3", "D1", "F1", "I1", "SP1"))
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    print("= PROYECTO PREDICCIÓN DE FUTBOL =\n")
    try:
        if args.action == "phase1":
            require("read_data")
            ejecutar_fase1()
        elif args.action == "phase2":
            require("train")
            ejecutar_fase2()
        elif args.action == "predict":
            require("predict")
            if not all((args.home, args.away, args.league)):
                parser.error("--home, --away and --league are required for predict")
            home_encoder = load_pkl("processed_data/le_home.pkl")
            away_encoder = load_pkl("processed_data/le_away.pkl")
            validate_match(
                args.home,
                args.away,
                args.league,
                set(home_encoder.classes_) | set(away_encoder.classes_),
                {"E0", "E1", "E2", "E3", "D1", "F1", "I1", "SP1"},
            )
            print("Match input accepted")
        else:
            require("read_data")
            require("train")
            ejecutar_fase1()
            ejecutar_fase2()
        return 0
    except (PermissionError, ValueError, RuntimeError, OSError) as exc:
        log.error("Pipeline failure: %s", type(exc).__name__)
        print("Operation failed. Check audit.log for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
