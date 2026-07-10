from __future__ import annotations
import sys
import time


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact H:MM:SS / MM:SS string."""
    if seconds < 0:
        seconds = 0
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours:d}:{minutes:02d}:{secs:02d}'
    return f'{minutes:02d}:{secs:02d}'


def _render_progress(
    current: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    bar_length: int = 40,
    elapsed: float | None = None,
) -> None:
    """Render a single progress-bar frame, optionally with an ETA."""
    percent = 100 * (current / float(total))
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    eta_str = ''
    if elapsed is not None and current > 0:
        remaining = elapsed * (total - current) / current
        eta_str = f' ETA {_format_duration(remaining)}'

    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}{eta_str}')
    sys.stdout.flush()

    if current == total:
        if elapsed is not None:
            sys.stdout.write(f' (elapsed {_format_duration(elapsed)})')
        sys.stdout.write('\n')
        sys.stdout.flush()


def print_progress(
    current: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    bar_length: int = 40,
    start_time: float | None = None,
) -> None:
    """
    Print a progress bar to stdout that updates in place.

    Parameters
    ----------
    current : int
        Current iteration (0-indexed)
    total : int
        Total iterations
    prefix : str
        Text to display before the progress bar
    suffix : str
        Text to display after the progress bar
    bar_length : int
        Length of the progress bar in characters
    start_time : float, optional
        If provided (e.g. a ``time.time()`` timestamp captured before the loop),
        an estimate of the remaining execution time is appended to the bar,
        extrapolated from the elapsed time and the current progress. When omitted
        no ETA is shown. For repeated calls in a loop, consider using
        :class:`ProgressBar` instead, which tracks the start time for you.
    """
    elapsed = None if start_time is None else time.time() - start_time
    _render_progress(current, total, prefix, suffix, bar_length, elapsed)


class ProgressBar:
    """
    Stateful progress bar that updates in place and can display an estimate of
    the remaining execution time.

    The start time is captured on construction (or reset via :meth:`reset`), so
    each :meth:`update` can extrapolate the remaining time from the elapsed time
    and the current progress.

    Example
    -------
    >>> progress = ProgressBar(Nsegs, prefix='Progress:', show_eta=True)
    >>> for i in range(Nsegs):
    ...     # ... do work ...
    ...     progress.update(i + 1, suffix=f'Block {i+1}/{Nsegs}')
    """

    def __init__(
        self,
        total: int,
        prefix: str = '',
        suffix: str = '',
        bar_length: int = 40,
        show_eta: bool = False,
    ) -> None:
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.show_eta = show_eta
        self.start_time = time.time()

    def reset(self) -> None:
        """Restart the elapsed-time measurement."""
        self.start_time = time.time()

    def update(
        self,
        current: int,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> None:
        """
        Redraw the bar for the given iteration.

        Parameters
        ----------
        current : int
            Current iteration.
        prefix, suffix : str, optional
            Override the prefix/suffix set on construction for this frame.
        """
        elapsed = time.time() - self.start_time if self.show_eta else None
        _render_progress(
            current,
            self.total,
            self.prefix if prefix is None else prefix,
            self.suffix if suffix is None else suffix,
            self.bar_length,
            elapsed,
        )
