# Sample images

Drop a food photo here as `plate.jpg` (or `.png`) and the image evals and the
image-path benchmark will pick it up:

    python -m evals.run_evals --image samples/plate.jpg
    python -m bench.latency --runs 12 --image samples/plate.jpg

A good sample for this project is an ordinary phone photo of a real plate --
ideally an Indian home-cooked one with two or three distinguishable items, since
that is what the nutrition table is weighted toward. A styled restaurant shot is
an easier case than the ones this agent actually has to handle.

These are committed so the image path is reproducible from a clean clone.
Keep them small (under ~1 MB); resize rather than commit a 5 MB original.

## About plate.jpg

An Indian thali: bajra roti, curd, a vegetable curry, a dry sabzi and two
laddus. It is a stock photograph (612x344, stock-style EXIF caption), not a
real user snapshot, and that matters when reading the vision numbers: it is
evenly lit, centred and unobstructed. Real WhatsApp food photos are worse --
half-eaten plates, bad light, a hand in frame, steam on the lens. The vision
accuracy reported in the README is therefore an optimistic bound, not a
representative one.

Anyone re-running these evals with their own photo should expect lower
confidence scores and more clarifying questions, which is the system working as
intended rather than a regression.
