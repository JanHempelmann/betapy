"""
Animated splash screen for betapy.

Shows the logo with a spinning dot-ring indicator and a status line
that updates as each component loads.
"""

import math
from pathlib import Path

try:
    from betapy import __version__ as _VERSION
    _VERSION = f'v{_VERSION}'
except Exception:
    _VERSION = ''

from PyQt5.QtWidgets import QSplashScreen, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen

# -----------------------------------------------------------------------
# Layout constants (pixels)
# -----------------------------------------------------------------------
LOGO_W      = 280    # logo display width
SPLASH_W    = 380    # total splash width
LOGO_PAD_T  = 24     # padding above logo
SPINNER_GAP = 36     # gap between logo bottom and spinner centre
SPINNER_R   = 20     # radius of the dot ring
DOT_R       = 4.2    # radius of each dot
N_DOTS      = 12
STATUS_GAP  = 12     # gap between spinner bottom and status text
STATUS_H    = 22
BOT_PAD     = 18     # padding below status text

# Timing
SPIN_STEP_DEG = 30   # degrees per tick
TIMER_MS      = 65   # ~15 fps

# Brand colours (matched to logo)
_NAVY  = (28,  52,  110)
_GREY  = (110, 110, 110)
_WHITE = (255, 255, 255)


class BetapySplashScreen(QSplashScreen):
    """
    Frameless splash screen with animated spinner and live status text.

    Usage
    -----
    splash = BetapySplashScreen()
    splash.show()
    app.processEvents()
    ...
    splash.set_status('Loading FORCE_CONSTANTS…')
    ...
    splash.finish(main_window)
    """

    def __init__(self):
        logo_path = Path(__file__).parent.parent / 'data' / 'logo.png'
        logo_pix = QPixmap(str(logo_path))
        if logo_pix.isNull():
            # Fallback: blank white pixmap so the splash still appears
            logo_pix = QPixmap(LOGO_W, LOGO_W)
            logo_pix.fill(QColor(*_WHITE))

        scaled   = logo_pix.scaledToWidth(LOGO_W, Qt.SmoothTransformation)
        logo_h   = scaled.height()

        splash_h = (LOGO_PAD_T + logo_h + SPINNER_GAP
                    + int(DOT_R) * 2 + STATUS_GAP + STATUS_H + BOT_PAD)

        base = QPixmap(SPLASH_W, splash_h)
        base.fill(QColor(*_WHITE))

        p = QPainter(base)
        p.drawPixmap((SPLASH_W - LOGO_W) // 2, LOGO_PAD_T, scaled)
        # Thin border so the white splash is visible on white desktops
        p.setPen(QPen(QColor(210, 210, 210), 1))
        p.drawRect(0, 0, SPLASH_W - 1, splash_h - 1)
        p.end()

        self._logo_bottom = LOGO_PAD_T + logo_h
        self._spin_angle  = 0
        self._status      = 'Starting…'

        super().__init__(base)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TIMER_MS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, message: str):
        """Update the status line and repaint immediately."""
        self._status = message
        self.repaint()
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick(self):
        self._spin_angle = (self._spin_angle - SPIN_STEP_DEG) % 360
        self.repaint()

    def drawContents(self, painter: QPainter):
        """Called by QSplashScreen.paintEvent after the base pixmap."""
        cx = self.width() // 2
        cy = self._logo_bottom + SPINNER_GAP

        # Spinner dots — head is most opaque, trail fades behind it
        for i in range(N_DOTS):
            angle_rad = math.radians(self._spin_angle + i * (360 / N_DOTS))
            x = cx + SPINNER_R * math.cos(angle_rad)
            y = cy + SPINNER_R * math.sin(angle_rad)
            # i=0 is the tail (faintest), i=N_DOTS-1 is the head (brightest)
            alpha = int(30 + 225 * (i / (N_DOTS - 1)) ** 1.5)
            r, g, b = _NAVY
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(r, g, b, alpha))
            painter.drawEllipse(QPointF(x, y), DOT_R, DOT_R)

        # Status text
        painter.setPen(QColor(*_GREY))
        font = QFont('Arial', 10)
        painter.setFont(font)
        text_y = cy + int(DOT_R) + STATUS_GAP
        painter.drawText(
            QRect(0, text_y, self.width(), STATUS_H),
            Qt.AlignHCenter | Qt.AlignVCenter,
            self._status,
        )

        # Version number — bottom right corner
        if _VERSION:
            painter.setPen(QColor(*_GREY))
            painter.setFont(QFont('Arial', 11))
            painter.drawText(
                QRect(0, self.height() - BOT_PAD - 4, self.width() - 8, BOT_PAD + 4),
                Qt.AlignRight | Qt.AlignVCenter,
                _VERSION,
            )
