import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import mediafile
from beets import ui
from beets.importer import ImportTask
from beets.library import Item, Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, should_write
from beets.util import syspath
from beets.dbcore import types

from vibenet import labels as FIELDS
from vibenet import load_model
from vibenet.core import load_audio


class VibeNetPlugin(BeetsPlugin):
    item_types = {f: types.NullFloat(6) for f in FIELDS}

    def __init__(self):
        super().__init__()

        self.config.add(
            {
                "threads": 0,
                "auto": True,
                "force": False,
                "skip_failed": True,
            }
        )

        self.cfg_threads = self.config["threads"].get(int)
        self.cfg_auto = self.config["auto"].get(bool)
        self.cfg_force = self.config["force"].get(bool)
        self.cfg_skip_failed = self.config["skip_failed"].get(bool)

        for name in FIELDS:
            field = mediafile.MediaField(
                mediafile.MP3DescStorageStyle(name), mediafile.StorageStyle(name)
            )
            self.add_media_field(name, field)

        self.import_stages = [self.imported]

    def _process_items(
        self,
        items: list[Item],
        threads: int,
        dry_run: bool,
        write_tags: bool,
        force: bool,
    ):
        if not force:
            # Skip items that already have tags
            items = [it for it in items if any(it.get(f) is None for f in FIELDS)]

        if not threads or threads == 0:
            threads = multiprocessing.cpu_count()
            self._log.debug("Adjusting max threads to CPU count: {}", threads)

        net = load_model()

        def worker(item) -> tuple[Item, dict]:
            try:
                path = syspath(item.path)

                # Basic file validation
                if not os.path.exists(path):
                    self._log.error("File not found: {}", path)
                    empty_scores = {f: 0.0 for f in FIELDS}
                    return item, empty_scores

                if os.path.getsize(path) == 0:
                    self._log.error("Empty file: {}", path)
                    empty_scores = {f: 0.0 for f in FIELDS}
                    return item, empty_scores

                wf = load_audio(path, 16000)
                pred = net.predict([wf], 16000)[0]
                scores = pred.to_dict()
                return item, scores
            except (OSError, IOError) as e:
                self._log.error("File I/O error processing {}: {}", item.path, e)
                empty_scores = {f: 0.0 for f in FIELDS}
                return item, empty_scores
            except (RuntimeError, ValueError) as e:
                self._log.error("Audio processing error for {}: {}", item.path, e)
                empty_scores = {f: 0.0 for f in FIELDS}
                return item, empty_scores
            except Exception as e:  # noqa: BLE001
                self._log.error(
                    "Unexpected error processing audio file {}: {}", item.path, e
                )
                # Return empty scores for failed items
                empty_scores = {f: 0.0 for f in FIELDS}
                return item, empty_scores

        total = len(items)
        finished = 0

        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(worker, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                idx = futs[fut]
                item = items[idx]

                try:
                    it, scores = fut.result()

                    # Skip items that had processing errors (empty scores)
                    if all(score == 0.0 for score in scores.values()):
                        if self.cfg_skip_failed:
                            self._log.warning(
                                "Skipping {} due to processing error", item.path
                            )
                            finished += 1
                            continue
                        else:
                            self._log.warning(
                                "Processing failed for {}, storing zero values",
                                item.path,
                            )

                except Exception as e:  # noqa: BLE001
                    self._log.error(
                        "Unexpected error processing {}: {}",
                        item.path,
                        e,
                        exc_info=True,
                    )
                    finished += 1
                    continue

                if not dry_run:
                    for f in FIELDS:
                        it[f] = float(scores[f])

                    it.store()

                    if write_tags:
                        it.write()

                finished += 1
                self._log.info(
                    "Progress: [{}/{}] ({} - {} - {})",
                    str(finished),
                    str(total),
                    it.artist,
                    it.album,
                    it.title,
                )

    def commands(self):
        cmd = Subcommand(
            "vibenet", help="Predict VibeNet attributes and store them on items."
        )

        cmd.parser.add_option(
            "-d",
            "--dry-run",
            action="store_true",
            dest="dryrun",
            help="Run predictions but do not save results to the library or files.",
        )

        cmd.parser.add_option(
            "-w",
            "--write",
            action="store_true",
            dest="write",
            help="Also write predicted attributes into file tags (not just the beets database).",
        )

        cmd.parser.add_option(
            "-t",
            "--threads",
            action="store",
            dest="threads",
            type="int",
            default=self.cfg_threads,
            help="Number of worker threads to use for predictions.",
        )

        cmd.parser.add_option(
            "-f",
            "--force",
            action="store_true",
            dest="force",
            default=self.cfg_force,
            help="Recompute and overwrite attributes even if they are already present.",
        )
        cmd.func = self._run_cmd
        return [cmd]

    def imported(self, _, task: ImportTask) -> None:
        if not self.cfg_auto:
            return

        items = list(task.imported_items())
        self._process_items(
            items,
            threads=self.cfg_threads,
            dry_run=False,
            write_tags=should_write(),
            force=self.cfg_force,
        )

    def _run_cmd(self, lib: Library, opts, args):
        query = ui.decargs(args)
        items = lib.items(query)

        if opts.dryrun:
            self._log.warning("*******************************************")
            self._log.warning("*** DRY RUN: NO CHANGES WILL BE APPLIED ***")
            self._log.warning("*******************************************")

        self._process_items(
            items,
            threads=opts.threads,
            dry_run=opts.dryrun,
            write_tags=opts.write,
            force=opts.force,
        )
