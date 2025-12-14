from . import _dll

class KokkosRuntime:

    _kokkos_initialized = False
    _kokkos_finalized = False

    # ============= Kokkos ====================

    @property
    def kokkos_initialized(self):
        return self._kokkos_initialized

    @property
    def kokkos_finalized(self):
        return self._kokkos_finalized


    def kokkos_initialize(self):
        """Call before any operations are performed."""
        if self._kokkos_finalized:
            raise RuntimeError("Kokkos already finalized. Restart python to initialize kokkos again.")

        if self._kokkos_initialized:
            raise RuntimeWarning("Kokkos already initialized. Doing nothing.")

        _dll.kokkos_initialize()
        self._kokkos_initialized = True

    def kokkos_finalize(self):
        """Call after all operations are performed."""
        if self._kokkos_finalized:
            raise RuntimeWarning("Kokkos already finalized. Restart python to initialize kokkos again.")

        _dll.kokkos_finalize()
        self._kokkos_finalized = True

    def is_running(self):
        return self._kokkos_initialized and (not self._kokkos_finalized)

kokkos_runtime = KokkosRuntime()
