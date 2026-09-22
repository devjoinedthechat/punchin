# Security

punchin runs agents against a fake dealership system that lives entirely in a local JSON file. It makes
no network calls of its own; the only outbound traffic is whatever the model backend you choose makes.

Recordings are written to `.punchin/`, which is git-ignored. They contain the conversation, the tool
calls and the booking outcome. Recordings of real calls would contain personal data, so treat that
directory the way you treat the call archive it came from.

The pinned customer is built from an extracted goal state, not from the original audio or transcript, so
a fork can be run against a de-identified persona once the goal state has been reviewed.

To report a vulnerability, open a GitHub security advisory on the repository.
