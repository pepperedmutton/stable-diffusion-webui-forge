# Bundled runtime sources

These snapshots preserve the extensions and model detection code used by this
fork. The Colab launcher installs them into `extensions/` and
`repositories/huggingface_guess/` in the runtime checkout.

`manifest.json` records each upstream repository, its base commit, local changes,
and the SHA-256 of every bundled source file. Each package retains its own
license. Model weights, personal wildcards, cache files, Git metadata, sample
images, and notebook outputs are excluded. Upstream READMEs may refer to those
omitted examples; use their recorded upstream URLs to view them.

The Krea tokenizer files are required runtime data and are included. The Krea
backend and `huggingface_guess` snapshots contain local changes; replacing them
with an upstream checkout removes those changes.
