import math

def format_header(headline: str) -> str:
    """Return underlined headline."""
    msg = f"\n____________________________{headline:_<40}"
    return msg

def _ns_to_readable_time(nanoseconds: int) -> str:
    hours = math.floor(nanoseconds/(1e9*3600))
    minutes = math.floor( (nanoseconds - hours*1e9*3600)/(1e9*60) )
    seconds = (nanoseconds - minutes*1e9*60)/(1e9)
    return f"{hours}h {minutes:>2}min {seconds:>5.2f}s"

def format_runtime(runtimes: dict[str,float]) -> str:
    """Return formatted string of run times."""
    total_time = sum(runtimes.values())
    msg = format_header("RUN TIMES")
    for time in runtimes.keys():
            msg += f"\n{time:_^12}:\t {_ns_to_readable_time(runtimes[time])} | {100*runtimes[time]/total_time:5.02f}%"
    msg += f"\n___total____:\t {_ns_to_readable_time(sum(runtimes.values()))}"
    return msg