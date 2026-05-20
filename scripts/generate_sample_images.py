from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


OUTPUT_DIR = Path("data/raw")


def make_sample(path: Path, seed: int, base_color: tuple[int, int, int], spot_color: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (256, 256), base_color)
    draw = ImageDraw.Draw(image)

    for _ in range(18):
        x = int(rng.integers(35, 190))
        y = int(rng.integers(35, 190))
        rx = int(rng.integers(18, 48))
        ry = int(rng.integers(14, 42))
        color = tuple(
            int(np.clip(channel + rng.normal(0, 12), 0, 255))
            for channel in spot_color
        )
        draw.ellipse((x, y, x + rx, y + ry), fill=color)

    noise = rng.normal(0, 8, (256, 256, 3))
    array = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=0.6))
    result.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    palettes = [
        ((224, 177, 145), (109, 63, 54)),
        ((210, 162, 132), (145, 52, 71)),
        ((236, 194, 164), (88, 74, 107)),
    ]

    for group, palette in enumerate(palettes, start=1):
        for index in range(1, 5):
            make_sample(
                OUTPUT_DIR / f"sample_group_{group}_{index}.png",
                seed=group * 100 + index,
                base_color=palette[0],
                spot_color=palette[1],
            )

    print(f"Imagenes sinteticas creadas en {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
