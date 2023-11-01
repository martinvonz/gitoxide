pub use gix_diff::*;

/// A structure to capture how to perform rename and copy tracking.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Rewrites {
    /// If `Some(…)`, also find copies. `None` is the default which does not try to detect copies at all.
    ///
    /// Note that this is an even more expensive operation than detecting renames stemming from additions and deletions
    /// as the resulting set to search through is usually larger.
    pub copies: Option<rewrites::Copies>,
    /// The percentage of similarity needed for files to be considered renamed, defaulting to `Some(0.5)`.
    /// This field is similar to `git diff -M50%`.
    ///
    /// If `None`, files are only considered equal if their content matches 100%.
    /// Note that values greater than 1.0 have no different effect than 1.0.
    pub percentage: Option<f32>,
    /// The amount of files to consider for fuzzy rename or copy tracking. Defaults to 1000, meaning that only 1000*1000
    /// combinations can be tested for fuzzy matches, i.e. the ones that try to find matches by comparing similarity.
    /// If 0, there is no limit.
    ///
    /// If the limit would not be enough to test the entire set of combinations, the algorithm will trade in precision and not
    /// run the fuzzy version of identity tests at all. That way results are never partial.
    pub limit: usize,
}

///
// TODO: merge this into the existing blob module in `gix-diff`
pub mod blob {
    #[cfg(feature = "blob-diff")]
    pub use gix_diff::blob::*;
    /// Information about the diff performed to detect similarity.
    #[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
    pub struct DiffLineStats {
        /// The amount of lines to remove from the source to get to the destination.
        pub removals: u32,
        /// The amount of lines to add to the source to get to the destination.
        pub insertions: u32,
        /// The amount of lines of the previous state, in the source.
        pub before: u32,
        /// The amount of lines of the new state, in the destination.
        pub after: u32,
    }
}

///
pub mod rename {
    /// Determine how to do rename tracking.
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Tracking {
        /// Do not track renames at all, the fastest option.
        Disabled,
        /// Track renames.
        Renames,
        /// Track renames and copies.
        ///
        /// This is the most expensive option.
        RenamesAndCopies,
    }
}

///
pub mod rewrites {
    use crate::diff::Rewrites;

    /// A type to retain state related to an ongoing tracking operation to retain sets of interesting changes
    /// of which some are retained to at a later stage compute the ones that seem to be renames or copies.
    #[cfg(feature = "blob-diff")]
    pub struct Tracker {
        /// The tracked items thus far, which will be used to determine renames/copies and rewrites later.
        items: Vec<tracker::Item>,
        /// A place to store all paths in to reduce amount of allocations.
        path_backing: Vec<u8>,
        /// A buffer for use when fetching objects for similarity tests.
        buf1: Vec<u8>,
        /// Another buffer for use when fetching objects for similarity tests.
        buf2: Vec<u8>,
        /// How to track copies and/or rewrites.
        rewrites: Rewrites,
        /// The diff algorithm to use when checking for similarity.
        diff_algo: gix_diff::blob::Algorithm,
    }

    // TODO: move this to `gix_diff::tree::visit` where `Change` is defined, once all this moves to `gix_diff`.
    #[cfg(feature = "blob-diff")]
    mod change_impls {
        use crate::diff::rewrites::tracker::{ChangeKind, ItemEntryMode};
        use gix_diff::tree::visit::Change;
        use gix_hash::oid;
        use gix_object::tree::EntryMode;

        fn into_mode(mode: EntryMode) -> Option<ItemEntryMode> {
            Some(match mode {
                EntryMode::Blob | EntryMode::BlobExecutable => ItemEntryMode::Blob,
                EntryMode::Link => ItemEntryMode::Link,
                EntryMode::Tree | EntryMode::Commit => return None,
            })
        }

        impl crate::diff::rewrites::tracker::ChangeTrait for gix_diff::tree::visit::Change {
            fn id(&self) -> &oid {
                match self {
                    Change::Addition { oid, .. } | Change::Deletion { oid, .. } | Change::Modification { oid, .. } => {
                        oid
                    }
                }
            }

            fn kind(&self) -> ChangeKind {
                match self {
                    Change::Addition { .. } => ChangeKind::Addition,
                    Change::Deletion { .. } => ChangeKind::Deletion,
                    Change::Modification { .. } => ChangeKind::Modification,
                }
            }

            fn entry_mode(&self) -> Option<ItemEntryMode> {
                match self {
                    Change::Addition { entry_mode, .. }
                    | Change::Deletion { entry_mode, .. }
                    | Change::Modification { entry_mode, .. } => into_mode(*entry_mode),
                }
            }

            fn id_and_entry_mode(&self) -> (&oid, ItemEntryMode) {
                match self {
                    Change::Addition { entry_mode, oid, .. }
                    | Change::Deletion { entry_mode, oid, .. }
                    | Change::Modification { entry_mode, oid, .. } => {
                        (oid, into_mode(*entry_mode).expect("only called if mode is convertible"))
                    }
                }
            }
        }
    }

    /// Types related to the rename tracker for renames, rewrites and copies.
    // TODO: remove feature once in `gix-diff`.
    #[cfg(feature = "blob-diff")]
    pub mod tracker {
        use std::ops::Range;

        use gix_diff::tree::visit::Change;
        use gix_object::tree::EntryMode;

        use crate::diff::blob::DiffLineStats;
        use crate::diff::{rewrites::Tracker, Rewrites};
        use crate::{
            bstr::BStr,
            diff::rewrites::{CopySource, Outcome},
        };

        /// The kind of a change.
        #[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq)]
        pub enum ChangeKind {
            /// The change represents the *deletion* of an item.
            Deletion,
            /// The change represents the *modification* of an item.
            Modification,
            /// The change represents the *addition* of an item.
            Addition,
        }

        /// The kind of entry that the change contains.
        // TODO: rename into EntryMode
        pub enum ItemEntryMode {
            /// The item is just data.
            Blob,
            /// The item is a symbolic link.
            Link,
        }

        /// A trait providing all functionality to abstract over the concept of a change, as seen by the [`Tracker`].
        // TODO: rename to `Change`.
        pub trait ChangeTrait {
            /// Return the hash of this change for identification.
            fn id(&self) -> &gix_hash::oid;
            /// Return the kind of this change.
            fn kind(&self) -> ChangeKind;
            /// Return more information about the kind of entry affected by this change, as long as it matches
            /// a known mode.
            fn entry_mode(&self) -> Option<ItemEntryMode>;
            /// Return the id of the change along with its mode.
            /// Note that this can panic if [`Self::entry_mode`] would return `None`, but it won't be called unless
            /// it is `Some(…)`.
            fn id_and_entry_mode(&self) -> (&gix_hash::oid, ItemEntryMode);
        }

        /// A set of tracked items allows to figure out their relations by figuring out their similarity.
        pub struct Item {
            /// The underlying raw change
            change: Change,
            /// That slice into the backing for paths.
            path: Range<usize>,
            /// If true, this item was already emitted, i.e. seen by the caller.
            emitted: bool,
        }

        impl Item {
            fn location<'a>(&self, backing: &'a [u8]) -> &'a BStr {
                backing[self.path.clone()].as_ref()
            }
            fn entry_mode_compatible(&self, mode: EntryMode) -> bool {
                use EntryMode::*;
                matches!(
                    (mode, self.change.entry_mode()),
                    (Blob | BlobExecutable, Blob | BlobExecutable) | (Link, Link)
                )
            }

            fn is_source_for_destination_of(&self, kind: visit::Kind, dest_item_mode: EntryMode) -> bool {
                self.entry_mode_compatible(dest_item_mode)
                    && match kind {
                        visit::Kind::RenameTarget => !self.emitted && matches!(self.change, Change::Deletion { .. }),
                        visit::Kind::CopyDestination => {
                            matches!(self.change, Change::Modification { .. })
                        }
                    }
            }
        }

        ///
        pub(crate) mod visit {
            use crate::{bstr::BStr, diff::blob::DiffLineStats};

            pub struct Source<'a> {
                pub mode: gix_object::tree::EntryMode,
                pub id: gix_hash::ObjectId,
                pub kind: Kind,
                pub location: &'a BStr,
                pub diff: Option<DiffLineStats>,
            }

            #[derive(Debug, Copy, Clone, Eq, PartialEq)]
            pub enum Kind {
                RenameTarget,
                CopyDestination,
            }

            pub struct Destination<'a> {
                pub change: gix_diff::tree::visit::Change,
                pub location: &'a BStr,
            }
        }

        ///
        pub mod emit {
            /// The error returned by [Tracker::emit()](super::Tracker::emit()).
            #[derive(Debug, thiserror::Error)]
            #[allow(missing_docs)]
            pub enum Error {
                #[error("Could not find blob for similarity checking")]
                FindExistingBlob(#[source] Box<dyn std::error::Error + Send + Sync>),
                #[error("Could not obtain exhaustive item set to use as possible sources for copy detection")]
                GetItemsForExhaustiveCopyDetection(#[source] Box<dyn std::error::Error + Send + Sync>),
            }
        }

        /// Lifecycle
        impl Tracker {
            /// Create a new instance with `rewrites` configuration, and the `diff_algo` to use when performing
            /// similarity checking.
            pub fn new(rewrites: Rewrites, diff_algo: gix_diff::blob::Algorithm) -> Self {
                Tracker {
                    items: vec![],
                    path_backing: vec![],
                    buf1: Vec::new(),
                    buf2: Vec::new(),
                    rewrites,
                    diff_algo,
                }
            }
        }

        /// build state and find matches.
        impl Tracker {
            /// We may refuse the push if that information isn't needed for what we have to track.
            pub fn try_push_change(&mut self, change: Change, location: &BStr) -> Option<Change> {
                if !change.entry_mode().is_blob_or_symlink() {
                    return Some(change);
                }
                let keep = match (self.rewrites.copies, &change) {
                    (Some(_find_copies), _) => true,
                    (None, Change::Modification { .. }) => false,
                    (None, _) => true,
                };

                if !keep {
                    return Some(change);
                }

                let start = self.path_backing.len();
                self.path_backing.extend_from_slice(location);
                self.items.push(Item {
                    path: start..self.path_backing.len(),
                    change,
                    emitted: false,
                });
                None
            }

            /// Can only be called once effectively as it alters its own state.
            ///
            /// `cb(destination, source)` is called for each item, either with `Some(source)` if it's
            /// the destination of a copy or rename, or with `None` for source if no relation to other
            /// items in the tracked set exist.
            /// `find_blob(oid, buf) -> Result<BlobRef, E>` is used to access blob data for similarity checks
            /// if required with data and is taken directly from the object database. Worktree filters and diff conversions
            /// will be applied afterwards automatically.
            /// `push_source_tree(push_fn: push(change, location))` is a function that is called when the entire tree of the source
            /// should be added as modifications by calling `push` repeatedly to use for perfect copy tracking. Note that
            /// `push` will panic if `change` is not a modification, and it's valid to not call `push` at all.
            pub fn emit<FindFn, E1, PushSourceTreeFn, E2>(
                &mut self,
                mut cb: impl FnMut(visit::Destination<'_>, Option<visit::Source<'_>>) -> gix_diff::tree::visit::Action,
                mut find_blob: FindFn,
                mut push_source_tree: PushSourceTreeFn,
            ) -> Result<Outcome, emit::Error>
            where
                FindFn: for<'b> FnMut(&gix_hash::oid, &'b mut Vec<u8>) -> Result<gix_object::BlobRef<'b>, E1>,
                PushSourceTreeFn: FnMut(&mut dyn FnMut(Change, &BStr)) -> Result<(), E2>,
                E1: std::error::Error + Send + Sync + 'static,
                E2: std::error::Error + Send + Sync + 'static,
            {
                fn by_id_and_location(a: &Item, b: &Item) -> std::cmp::Ordering {
                    a.change
                        .oid()
                        .cmp(b.change.oid())
                        .then_with(|| a.path.start.cmp(&b.path.start).then(a.path.end.cmp(&b.path.end)))
                }
                self.items.sort_by(by_id_and_location);

                let mut out = Outcome {
                    options: self.rewrites,
                    ..Default::default()
                };
                out = self.match_pairs_of_kind(
                    visit::Kind::RenameTarget,
                    &mut cb,
                    self.rewrites.percentage,
                    out,
                    &mut find_blob,
                )?;

                if let Some(copies) = self.rewrites.copies {
                    out = self.match_pairs_of_kind(
                        visit::Kind::CopyDestination,
                        &mut cb,
                        copies.percentage,
                        out,
                        &mut find_blob,
                    )?;

                    match copies.source {
                        CopySource::FromSetOfModifiedFiles => {}
                        CopySource::FromSetOfModifiedFilesAndAllSources => {
                            push_source_tree(&mut |change, location| {
                                assert!(
                                    self.try_push_change(change, location).is_none(),
                                    "we must accept every change"
                                );
                                // make sure these aren't viable to be emitted anymore.
                                self.items.last_mut().expect("just pushed").emitted = true;
                            })
                            .map_err(|err| emit::Error::GetItemsForExhaustiveCopyDetection(Box::new(err)))?;
                            self.items.sort_by(by_id_and_location);

                            out = self.match_pairs_of_kind(
                                visit::Kind::CopyDestination,
                                &mut cb,
                                copies.percentage,
                                out,
                                &mut find_blob,
                            )?;
                        }
                    }
                }

                self.items
                    .sort_by(|a, b| a.location(&self.path_backing).cmp(b.location(&self.path_backing)));
                for item in self.items.drain(..).filter(|item| !item.emitted) {
                    if cb(
                        visit::Destination {
                            location: item.location(&self.path_backing),
                            change: item.change,
                        },
                        None,
                    ) == gix_diff::tree::visit::Action::Cancel
                    {
                        break;
                    }
                }
                Ok(out)
            }

            fn match_pairs_of_kind<FindFn, E>(
                &mut self,
                kind: visit::Kind,
                cb: &mut impl FnMut(visit::Destination<'_>, Option<visit::Source<'_>>) -> gix_diff::tree::visit::Action,
                percentage: Option<f32>,
                mut out: Outcome,
                mut find_blob: FindFn,
            ) -> Result<Outcome, emit::Error>
            where
                FindFn: for<'b> FnMut(&gix_hash::oid, &'b mut Vec<u8>) -> Result<gix_object::BlobRef<'b>, E>,
                E: std::error::Error + Send + Sync + 'static,
            {
                // we try to cheaply reduce the set of possibilities first, before possibly looking more exhaustively.
                let needs_second_pass = !needs_exact_match(percentage);
                if self.match_pairs(cb, None /* by identity */, kind, &mut out, &mut find_blob)?
                    == gix_diff::tree::visit::Action::Cancel
                {
                    return Ok(out);
                }
                if needs_second_pass {
                    let is_limited = if self.rewrites.limit == 0 {
                        false
                    } else if let Some(permutations) = permutations_over_limit(&self.items, self.rewrites.limit, kind) {
                        match kind {
                            visit::Kind::RenameTarget => {
                                out.num_similarity_checks_skipped_for_rename_tracking_due_to_limit = permutations;
                            }
                            visit::Kind::CopyDestination => {
                                out.num_similarity_checks_skipped_for_copy_tracking_due_to_limit = permutations;
                            }
                        }
                        true
                    } else {
                        false
                    };
                    if !is_limited {
                        self.match_pairs(cb, percentage, kind, &mut out, &mut find_blob)?;
                    }
                }
                Ok(out)
            }

            fn match_pairs<FindFn, E>(
                &mut self,
                cb: &mut impl FnMut(visit::Destination<'_>, Option<visit::Source<'_>>) -> gix_diff::tree::visit::Action,
                percentage: Option<f32>,
                kind: visit::Kind,
                stats: &mut Outcome,
                mut find_blob: FindFn,
            ) -> Result<gix_diff::tree::visit::Action, emit::Error>
            where
                FindFn: for<'b> FnMut(&gix_hash::oid, &'b mut Vec<u8>) -> Result<gix_object::BlobRef<'b>, E>,
                E: std::error::Error + Send + Sync + 'static,
            {
                // TODO(perf): reuse object data and interner state and interned tokens, make these available to `find_match()`
                let mut dest_ofs = 0;
                while let Some((mut dest_idx, dest)) =
                    self.items[dest_ofs..].iter().enumerate().find_map(|(idx, item)| {
                        (!item.emitted && matches!(item.change, Change::Addition { .. })).then_some((idx, item))
                    })
                {
                    dest_idx += dest_ofs;
                    dest_ofs = dest_idx + 1;
                    let src = find_match(
                        &self.items,
                        dest,
                        dest_idx,
                        percentage.map(|p| (p, self.diff_algo)),
                        kind,
                        stats,
                        &mut find_blob,
                        &mut self.buf1,
                        &mut self.buf2,
                    )?
                    .map(|(src_idx, src, diff)| {
                        let (id, mode) = src.change.oid_and_entry_mode();
                        let id = id.to_owned();
                        let location = src.location(&self.path_backing);
                        (
                            visit::Source {
                                mode,
                                id,
                                kind,
                                location,
                                diff,
                            },
                            src_idx,
                        )
                    });
                    if src.is_none() {
                        continue;
                    }
                    let location = dest.location(&self.path_backing);
                    let change = dest.change.clone();
                    let dest = visit::Destination { change, location };
                    self.items[dest_idx].emitted = true;
                    if let Some(src_idx) = src.as_ref().map(|t| t.1) {
                        self.items[src_idx].emitted = true;
                    }
                    if cb(dest, src.map(|t| t.0)) == gix_diff::tree::visit::Action::Cancel {
                        return Ok(gix_diff::tree::visit::Action::Cancel);
                    }
                }
                Ok(gix_diff::tree::visit::Action::Continue)
            }
        }

        fn permutations_over_limit(items: &[Item], limit: usize, kind: visit::Kind) -> Option<usize> {
            let (sources, destinations) = items
                .iter()
                .filter(|item| match kind {
                    visit::Kind::RenameTarget => !item.emitted,
                    visit::Kind::CopyDestination => true,
                })
                .fold((0, 0), |(mut src, mut dest), item| {
                    match item.change {
                        Change::Addition { .. } => {
                            dest += 1;
                        }
                        Change::Deletion { .. } => {
                            if kind == visit::Kind::RenameTarget {
                                src += 1
                            }
                        }
                        Change::Modification { .. } => {
                            if kind == visit::Kind::CopyDestination {
                                src += 1
                            }
                        }
                    }
                    (src, dest)
                });
            let permutations = sources * destinations;
            (permutations > limit * limit).then_some(permutations)
        }

        fn needs_exact_match(percentage: Option<f32>) -> bool {
            percentage.map_or(true, |p| p >= 1.0)
        }

        /// <`src_idx`, src, possibly diff stat>
        type SourceTuple<'a> = (usize, &'a Item, Option<DiffLineStats>);

        /// Find `item` in our set of items ignoring `item_idx` to avoid finding ourselves, by similarity indicated by `percentage`.
        /// The latter can be `None` or `Some(x)` where `x>=1` for identity, and anything else for similarity.
        /// We also ignore emitted items entirely.
        /// Use `kind` to indicate what kind of match we are looking for, which might be deletions matching an `item` addition, or
        /// any non-deletion otherwise.
        /// Note that we always try to find by identity first even if a percentage is given as it's much faster and may reduce the set
        /// of items to be searched.
        #[allow(clippy::too_many_arguments)]
        fn find_match<'a, FindFn, E>(
            items: &'a [Item],
            item: &Item,
            item_idx: usize,
            percentage: Option<(f32, gix_diff::blob::Algorithm)>,
            kind: visit::Kind,
            stats: &mut Outcome,
            mut find_worktree_blob: FindFn,
            buf1: &mut Vec<u8>,
            buf2: &mut Vec<u8>,
        ) -> Result<Option<SourceTuple<'a>>, emit::Error>
        where
            FindFn: for<'b> FnMut(&gix_hash::oid, &'b mut Vec<u8>) -> Result<gix_object::BlobRef<'b>, E>,
            E: std::error::Error + Send + Sync + 'static,
        {
            let (item_id, item_mode) = item.change.oid_and_entry_mode();
            if needs_exact_match(percentage.map(|t| t.0)) || item_mode == gix_object::tree::EntryMode::Link {
                let first_idx = items.partition_point(|a| a.change.oid() < item_id);
                let range = match items.get(first_idx..).map(|items| {
                    let end = items
                        .iter()
                        .position(|a| a.change.oid() != item_id)
                        .map_or(items.len(), |idx| first_idx + idx);
                    first_idx..end
                }) {
                    Some(range) => range,
                    None => return Ok(None),
                };
                if range.is_empty() {
                    return Ok(None);
                }
                let res = items[range.clone()].iter().enumerate().find_map(|(mut src_idx, src)| {
                    src_idx += range.start;
                    (src_idx != item_idx && src.is_source_for_destination_of(kind, item_mode))
                        .then_some((src_idx, src, None))
                });
                if let Some(src) = res {
                    return Ok(Some(src));
                }
            } else {
                let new =
                    find_worktree_blob(item_id, buf1).map_err(|err| emit::Error::FindExistingBlob(Box::new(err)))?;
                let (percentage, algo) = percentage.expect("it's set to something below 1.0 and we assured this");
                debug_assert!(
                    item.change.entry_mode().is_blob(),
                    "symlinks are matched exactly, and trees aren't used here"
                );
                for (can_idx, src) in items
                    .iter()
                    .enumerate()
                    .filter(|(src_idx, src)| *src_idx != item_idx && src.is_source_for_destination_of(kind, item_mode))
                {
                    let old = find_worktree_blob(src.change.oid(), buf2)
                        .map_err(|err| emit::Error::FindExistingBlob(Box::new(err)))?;
                    // TODO: make sure we get attribute handling and binary skips and filters right here. There is crate::object::blob::diff::Platform
                    //       which should have facilities for that one day, but we don't use it because we need newlines in our tokens.
                    let tokens = gix_diff::blob::intern::InternedInput::new(
                        gix_diff::blob::sources::byte_lines_with_terminator(old.data),
                        gix_diff::blob::sources::byte_lines_with_terminator(new.data),
                    );
                    let counts = gix_diff::blob::diff(
                        algo,
                        &tokens,
                        gix_diff::blob::sink::Counter::new(diff::Statistics {
                            removed_bytes: 0,
                            input: &tokens,
                        }),
                    );
                    let similarity =
                        (old.data.len() - counts.wrapped) as f32 / old.data.len().max(new.data.len()) as f32;
                    stats.num_similarity_checks += 1;
                    if similarity >= percentage {
                        return Ok(Some((
                            can_idx,
                            src,
                            DiffLineStats {
                                removals: counts.removals,
                                insertions: counts.insertions,
                                before: tokens.before.len().try_into().expect("interner handles only u32"),
                                after: tokens.after.len().try_into().expect("interner handles only u32"),
                            }
                            .into(),
                        )));
                    }
                }
            }
            Ok(None)
        }

        mod diff {
            use std::ops::Range;

            pub struct Statistics<'a, 'data> {
                pub removed_bytes: usize,
                pub input: &'a gix_diff::blob::intern::InternedInput<&'data [u8]>,
            }

            impl<'a, 'data> gix_diff::blob::Sink for Statistics<'a, 'data> {
                type Out = usize;

                fn process_change(&mut self, before: Range<u32>, _after: Range<u32>) {
                    self.removed_bytes = self.input.before[before.start as usize..before.end as usize]
                        .iter()
                        .map(|token| self.input.interner[*token].len())
                        .sum();
                }

                fn finish(self) -> Self::Out {
                    self.removed_bytes
                }
            }
        }
    }

    /// Determine in which set of files to search for copies.
    #[derive(Default, Debug, Copy, Clone, Eq, PartialEq)]
    pub enum CopySource {
        /// Find copies from the set of modified files only.
        #[default]
        FromSetOfModifiedFiles,
        /// Find copies from the set of modified files, as well as all files known to the source (i.e. previous state of the tree).
        ///
        /// This can be an expensive operation as it scales exponentially with the total amount of files in the set.
        FromSetOfModifiedFilesAndAllSources,
    }

    /// Under which circumstances we consider a file to be a copy.
    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Copies {
        /// The set of files to search when finding the source of copies.
        pub source: CopySource,
        /// Equivalent to [`Rewrites::percentage`], but used for copy tracking.
        ///
        /// Useful to have similarity-based rename tracking and cheaper copy tracking.
        pub percentage: Option<f32>,
    }

    impl Default for Copies {
        fn default() -> Self {
            Copies {
                source: CopySource::default(),
                percentage: Some(0.5),
            }
        }
    }

    /// Information collected while handling rewrites of files which may be tracked.
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Outcome {
        /// The options used to guide the rewrite tracking. Either fully provided by the caller or retrieved from git configuration.
        pub options: Rewrites,
        /// The amount of similarity checks that have been conducted to find renamed files and potentially copies.
        pub num_similarity_checks: usize,
        /// Set to the amount of worst-case rename permutations we didn't search as our limit didn't allow it.
        pub num_similarity_checks_skipped_for_rename_tracking_due_to_limit: usize,
        /// Set to the amount of worst-case copy permutations we didn't search as our limit didn't allow it.
        pub num_similarity_checks_skipped_for_copy_tracking_due_to_limit: usize,
    }

    /// The default settings for rewrites according to the git configuration defaults.
    impl Default for Rewrites {
        fn default() -> Self {
            Rewrites {
                copies: None,
                percentage: Some(0.5),
                limit: 1000,
            }
        }
    }

    #[cfg(feature = "blob-diff")]
    mod config {
        use crate::config::cache::util::ApplyLeniency;
        use crate::config::tree::Diff;
        use crate::diff::rename::Tracking;
        use crate::diff::rewrites::Copies;
        use crate::diff::Rewrites;

        /// The error returned by [`Rewrites::try_from_config()`].
        #[derive(Debug, thiserror::Error)]
        #[allow(missing_docs)]
        pub enum Error {
            #[error(transparent)]
            ConfigDiffRenames(#[from] crate::config::key::GenericError),
            #[error(transparent)]
            ConfigDiffRenameLimit(#[from] crate::config::unsigned_integer::Error),
        }

        impl Rewrites {
            /// Create an instance by reading all relevant information from the `config`uration, while being `lenient` or not.
            /// Returns `Ok(None)` if nothing is configured.
            ///
            /// Note that missing values will be defaulted similar to what git does.
            #[allow(clippy::result_large_err)]
            pub fn try_from_config(config: &gix_config::File<'static>, lenient: bool) -> Result<Option<Self>, Error> {
                let key = "diff.renames";
                let copies = match config
                    .boolean_by_key(key)
                    .map(|value| Diff::RENAMES.try_into_renames(value))
                    .transpose()
                    .with_leniency(lenient)?
                {
                    Some(renames) => match renames {
                        Tracking::Disabled => return Ok(None),
                        Tracking::Renames => None,
                        Tracking::RenamesAndCopies => Some(Copies::default()),
                    },
                    None => return Ok(None),
                };

                let default = Self::default();
                Ok(Rewrites {
                    copies,
                    limit: config
                        .integer_by_key("diff.renameLimit")
                        .map(|value| Diff::RENAME_LIMIT.try_into_usize(value))
                        .transpose()
                        .with_leniency(lenient)?
                        .unwrap_or(default.limit),
                    ..default
                }
                .into())
            }
        }
    }
    #[cfg(feature = "blob-diff")]
    pub use config::Error;
}
