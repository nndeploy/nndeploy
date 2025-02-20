from flet import colors, Theme, ThemeMode, LinearGradient, alignment

class ThemeManager:
    @staticmethod
    def get_theme():
        return {
            "light": {
                "background": colors.WHITE,
                "card_background": colors.WHITE,
                "text": "#1d1c1c",
                "border": colors.BLUE_GREY_300,
                "node_header": colors.BLUE_GREY_100,
                "toolbar": "#cccccc",
            },
            "dark": {
                "background": "#292e3b",
                "card_background": colors.BLUE_GREY_800,
                "text": colors.WHITE,
                "border": colors.BLUE_GREY_700,
                "node_header": colors.BLUE_GREY_700,
                "toolbar": LinearGradient(
                    begin=alignment.center_left,
                    end=alignment.center_right,
                    colors=["#56647b", "#b4c2dc"]
                ),
            }
        }

    @staticmethod
    def get_theme_mode(is_dark: bool):
        return ThemeMode.DARK if is_dark else ThemeMode.LIGHT