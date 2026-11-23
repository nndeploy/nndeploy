"""Immutable source video encoding contract for FFmpeg output."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping


def _optional_string(value: object) -> str | None:
    return None if value in (None, "", "N/A") else str(value)


@dataclass(frozen=True)
class VideoEncodingProfile:
    """Immutable origin-compatible FFmpeg output contract."""

    width: int
    height: int
    frame_rate: str
    encoder: str
    output_pix_fmt: str
    video_bitrate: str
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_range: str | None
    field_order: str

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> "VideoEncodingProfile":
        codec = metadata.get("codec_name")
        encoder_map = {"h264": "libx264", "prores": "prores_ks"}
        if not isinstance(codec, str) or codec.lower() not in encoder_map:
            raise ValueError(f"unsupported source codec: {codec!r}")

        frame_rate = metadata.get("fps_fraction")
        if not isinstance(frame_rate, str) or (
            "/" not in frame_rate and not frame_rate.isdecimal()
        ):
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}")
        try:
            fraction = Fraction(frame_rate)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}") from exc
        if fraction <= 0:
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}")

        raw_field_order = metadata.get("field_order")
        field_order_map = {
            None: "progressive",
            "progressive": "progressive",
            "top": "top_field_first",
            "top_field_first": "top_field_first",
            "tb": "top_field_first",
            "tt": "top_field_first",
            "tff": "top_field_first",
            "bottom": "bottom_field_first",
            "bottom_field_first": "bottom_field_first",
            "bb": "bottom_field_first",
            "bt": "bottom_field_first",
            "bff": "bottom_field_first",
            "interlaced": "interlaced",
        }
        if raw_field_order not in field_order_map:
            raise ValueError(f"unsupported field_order: {raw_field_order!r}")

        bitrate = metadata.get("bit_rate")
        video_bitrate = "10000000" if bitrate in (None, "", "N/A") else str(bitrate)
        return cls(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            frame_rate=f"{fraction.numerator}/{fraction.denominator}",
            encoder=encoder_map[codec.lower()],
            output_pix_fmt=str(metadata.get("pix_fmt") or "yuv420p"),
            video_bitrate=video_bitrate,
            color_space=_optional_string(metadata.get("color_space")),
            color_transfer=_optional_string(metadata.get("color_transfer")),
            color_primaries=_optional_string(metadata.get("color_primaries")),
            color_range=_optional_string(metadata.get("color_range")),
            field_order=field_order_map[raw_field_order],
        )

    def ffmpeg_output_args(self) -> tuple[str, ...]:
        args = [
            "-vcodec",
            self.encoder,
            "-b:v",
            self.video_bitrate,
            "-pix_fmt",
            self.output_pix_fmt,
            "-r",
            self.frame_rate,
        ]
        for option, value in (
            ("-colorspace", self.color_space),
            ("-color_trc", self.color_transfer),
            ("-color_primaries", self.color_primaries),
            ("-color_range", self.color_range),
        ):
            if value is not None:
                args.extend((option, value))
        if self.field_order == "top_field_first":
            args.extend(("-flags", "+ildct+ilme", "-vf", "setfield=tff"))
        elif self.field_order == "bottom_field_first":
            args.extend(("-flags", "+ildct+ilme", "-vf", "setfield=bff"))
        elif self.field_order == "interlaced":
            args.extend(("-flags", "+ildct+ilme"))
        return tuple(args)
