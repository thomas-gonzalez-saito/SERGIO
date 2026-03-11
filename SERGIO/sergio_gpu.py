"""
GPU-accelerated SERGIO simulator.

Provides a drop-in replacement for the original `sergio` class that vectorizes
the Chemical Langevin Equation (CLE) solver across all genes within each
network layer.  Uses CuPy when a CUDA GPU is available, otherwise falls back
transparently to NumPy (still benefiting from the vectorized formulation).
"""

import sys
import csv
import numpy as np

# ---------------------------------------------------------------------------
# Backend selection: prefer CuPy (GPU), fall back to NumPy (CPU)
# ---------------------------------------------------------------------------
try:
    import cupy as xp
    _HAS_GPU = True
    print("[sergio_gpu] CuPy detected — using GPU acceleration")
except ImportError:
    import numpy as xp
    _HAS_GPU = False
    print("[sergio_gpu] CuPy not found — falling back to NumPy (CPU)")


def _to_numpy(arr):
    """Convert an xp array to a NumPy array (no-op when xp *is* numpy)."""
    if _HAS_GPU:
        return xp.asnumpy(arr)
    return np.asarray(arr)


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════

class sergio_gpu:
    """Vectorised SERGIO steady-state simulator (CuPy / NumPy backend)."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        number_genes,
        number_bins,
        number_sc,
        noise_params,
        noise_type,
        decays,
        sampling_state=10,
        dt=0.01,
        optimize_sampling=False,
        # The following original kwargs are accepted for API compat but unused
        # in the static-simulation path.
        dynamics=False,
        tol=1e-3,
        window_length=100,
        bifurcation_matrix=None,
        noise_params_splice=None,
        noise_type_splice=None,
        splice_ratio=4,
        dt_splice=0.01,
        migration_rate=None,
    ):
        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.sampling_state_ = sampling_state
        self.dt_ = dt
        self.optimize_sampling_ = optimize_sampling
        self.noiseType_ = noise_type
        self.maxLevels_ = 0

        # Decay vector  (1-D, length nGenes)
        if np.isscalar(decays):
            self.decayVector_ = xp.asarray(np.repeat(decays, number_genes), dtype=xp.float64)
        else:
            self.decayVector_ = xp.asarray(np.asarray(decays, dtype=np.float64))

        # Noise amplitude vector  (1-D, length nGenes)
        if np.isscalar(noise_params):
            self.noiseParamsVector_ = xp.asarray(np.repeat(noise_params, number_genes), dtype=xp.float64)
        else:
            self.noiseParamsVector_ = xp.asarray(np.asarray(noise_params, dtype=np.float64))

        # Will be filled by build_graph / simulate
        self.graph_ = {}
        self.master_regulators_idx_ = set()
        self.level2geneIDs_ = {}          # level -> list of gene IDs
        self.gID_to_level_ = {}           # geneID -> level
        self.meanExpression = xp.full((number_genes, number_bins), -1.0, dtype=xp.float64)
        self.scExpressions_ = {}          # geneID -> (nBins, nSC) array

    # -------------------------------------------------------------- build_graph
    def build_graph(self, input_file_taregts, input_file_regs, shared_coop_state=0):
        """Parse target/regulator files – identical semantics to the original."""
        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []

        allTargets = []

        # --- Targets file ---------------------------------------------------
        with open(input_file_taregts, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            if shared_coop_state <= 0:
                for row in reader:
                    nRegs = int(row[1])
                    if nRegs == 0:
                        print("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                    currInteraction = []
                    currParents = []
                    for regId, K, C_state in zip(
                        row[2:2+nRegs],
                        row[2+nRegs:2+2*nRegs],
                        row[2+2*nRegs:2+3*nRegs],
                    ):
                        currInteraction.append(
                            (int(regId), float(K), float(C_state), 0.0)
                        )
                        currParents.append(int(regId))
                        self.graph_[int(regId)]['targets'].append(int(row[0]))

                    gid = int(row[0])
                    self.graph_[gid]['params'] = currInteraction
                    self.graph_[gid]['regs'] = currParents
                    self.graph_[gid]['level'] = -1
                    allTargets.append(gid)
            else:
                for row in reader:
                    nRegs = int(float(row[1]))
                    if nRegs == 0:
                        print("Error: a master regulator (#Regs = 0) appeared in input")
                        sys.exit()
                    currInteraction = []
                    currParents = []
                    for regId, K in zip(
                        row[2:2+nRegs],
                        row[2+nRegs:2+2*nRegs],
                    ):
                        currInteraction.append(
                            (int(float(regId)), float(K), shared_coop_state, 0.0)
                        )
                        currParents.append(int(float(regId)))
                        self.graph_[int(float(regId))]['targets'].append(int(float(row[0])))

                    gid = int(float(row[0]))
                    self.graph_[gid]['params'] = currInteraction
                    self.graph_[gid]['regs'] = currParents
                    self.graph_[gid]['level'] = -1
                    allTargets.append(gid)

        # --- Regulators file ------------------------------------------------
        with open(input_file_regs, 'r') as f:
            masterRegs = []
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if np.shape(row)[0] != self.nBins_ + 1:
                    print("Error: Inconsistent number of bins")
                    sys.exit()
                mrID = int(float(row[0]))
                masterRegs.append(mrID)
                self.graph_[mrID]['rates'] = [float(v) for v in row[1:]]
                self.graph_[mrID]['regs'] = []
                self.graph_[mrID]['level'] = -1

        self.master_regulators_idx_ = set(masterRegs)
        if len(self.master_regulators_idx_) + len(allTargets) != self.nGenes_:
            print("Error: Inconsistent number of genes")
            sys.exit()

        self._find_levels()
        self._set_sc_indices()

    # -------------------------------------------------------------- layering
    def _find_levels(self):
        """Bottom-up longest-path layering (same algorithm as original)."""
        U = set()
        Z = set()
        V = set(self.graph_.keys())

        currLayer = 0
        self.level2geneIDs_[currLayer] = []

        while U != V:
            currVerts = set(
                filter(lambda v: set(self.graph_[v]['targets']).issubset(Z), V - U)
            )
            for v in currVerts:
                self.graph_[v]['level'] = currLayer
                U.add(v)
                self.level2geneIDs_.setdefault(currLayer, []).append(v)
                self.gID_to_level_[v] = currLayer

            currLayer += 1
            Z = Z.union(U)
            self.level2geneIDs_[currLayer] = []

        self.level2geneIDs_.pop(currLayer)
        self.maxLevels_ = currLayer - 1

    def _set_sc_indices(self, safety_steps=0):
        if self.optimize_sampling_:
            state = np.true_divide(30000 - safety_steps * self.maxLevels_, self.nSC_)
            if state < self.sampling_state_:
                self.sampling_state_ = state
        self.scIndices_ = np.random.randint(
            low=-self.sampling_state_ * self.nSC_, high=0, size=self.nSC_
        )

    def _calculate_required_steps(self, level, safety_steps=0):
        return int(self.sampling_state_ * self.nSC_ + level * safety_steps)

    # ═══════════════════════════════════════════════════════════════════════
    # Vectorised CLE simulation
    # ═══════════════════════════════════════════════════════════════════════

    def simulate(self):
        for level in range(self.maxLevels_, -1, -1):
            print(f"[GPU] Start simulating level {level}")
            self._simulate_layer(level)
            print(f"[GPU] Done with level {level}")

    def _simulate_layer(self, level):
        geneIDs = self.level2geneIDs_[level]
        nGenes_layer = len(geneIDs)
        nBins = self.nBins_
        nReqSteps = self._calculate_required_steps(level)

        # Separate master regulators and target genes in this layer
        mr_mask = []
        target_mask = []
        for idx, gid in enumerate(geneIDs):
            if gid in self.master_regulators_idx_:
                mr_mask.append(idx)
            else:
                target_mask.append(idx)

        mr_mask = np.array(mr_mask, dtype=np.intp)
        target_mask = np.array(target_mask, dtype=np.intp)

        # ---- Half-response computation (uses meanExpression of regulators) ---
        # For each target gene, update the half-response (4th element in params tuple)
        for gid in geneIDs:
            if gid not in self.master_regulators_idx_:
                params = self.graph_[gid]['params']
                for c, interTuple in enumerate(params):
                    regIdx = interTuple[0]
                    meanArr = _to_numpy(self.meanExpression[regIdx])
                    if np.all(meanArr == -1):
                        print("Error: Expression of one or more genes in previous layer was not modeled.")
                        sys.exit()
                    self.graph_[gid]['params'][c] = (
                        interTuple[0], interTuple[1], interTuple[2], float(np.mean(meanArr))
                    )

        # ---- Build arrays that describe the regulation structure ------------
        # For target genes we need:
        #   reg_gene_ids[i]  : list of regulator gene IDs for target i
        #   K_vals[i]        : corresponding |K| values
        #   coop_states[i]   : cooperativity exponents
        #   half_responses[i]: half-response values
        #   repressive[i]    : boolean flags

        # We will build a *padded* representation so we can batch all targets.
        max_regs = 0
        target_gene_ids = [geneIDs[i] for i in target_mask]
        for gid in target_gene_ids:
            max_regs = max(max_regs, len(self.graph_[gid]['params']))

        nTargets = len(target_mask)

        if nTargets > 0 and max_regs > 0:
            # Padded arrays:  nTargets × max_regs
            reg_ids_padded = np.zeros((nTargets, max_regs), dtype=np.intp)
            K_abs = np.zeros((nTargets, max_regs), dtype=np.float64)
            coop = np.zeros((nTargets, max_regs), dtype=np.float64)
            h_resp = np.zeros((nTargets, max_regs), dtype=np.float64)
            is_repressive = np.zeros((nTargets, max_regs), dtype=np.bool_)
            n_regs_per_gene = np.zeros(nTargets, dtype=np.intp)
            # Mask for valid (non-padded) regulator slots
            reg_mask = np.zeros((nTargets, max_regs), dtype=np.bool_)

            for ti, gid in enumerate(target_gene_ids):
                params = self.graph_[gid]['params']
                nr = len(params)
                n_regs_per_gene[ti] = nr
                for ri, (rId, K, cs, hr) in enumerate(params):
                    reg_ids_padded[ti, ri] = rId
                    K_abs[ti, ri] = abs(K)
                    coop[ti, ri] = cs
                    h_resp[ti, ri] = hr
                    is_repressive[ti, ri] = (K < 0)
                    reg_mask[ti, ri] = True

            # Move to device
            K_abs_d = xp.asarray(K_abs)
            coop_d = xp.asarray(coop)
            h_resp_d = xp.asarray(h_resp)
            is_repressive_d = xp.asarray(is_repressive)
            reg_mask_d = xp.asarray(reg_mask)  # (nTargets, max_regs)
        else:
            nTargets = 0  # All genes in this layer are master regulators

        # ---- Initial concentrations: shape (nGenes_layer, nBins) -----------
        conc = xp.zeros((nGenes_layer, nBins), dtype=xp.float64)

        # Master regulators: rate / decay
        for local_idx in mr_mask:
            gid = geneIDs[local_idx]
            rates = xp.asarray(self.graph_[gid]['rates'], dtype=xp.float64)
            conc[local_idx, :] = rates / self.decayVector_[gid]

        # Target genes: sum(|K| * hill(meanExpr_reg, hr, cs)) / decay
        for ti, local_idx in enumerate(target_mask):
            gid = geneIDs[local_idx]
            params = self.graph_[gid]['params']
            for bIdx in range(nBins):
                rate = 0.0
                for (rId, K, cs, hr) in params:
                    me = float(_to_numpy(self.meanExpression[rId, bIdx]))
                    rate += abs(K) * self._hill_scalar(me, hr, cs, K < 0)
                conc[local_idx, bIdx] = rate / float(_to_numpy(self.decayVector_[gid]))

        # ---- Allocate concentration history buffer -------------------------
        # We only need to keep the full trajectory for sampling at the end.
        # Shape: (nReqSteps, nGenes_layer, nBins)
        conc_history = xp.zeros((nReqSteps, nGenes_layer, nBins), dtype=xp.float64)
        conc_history[0] = conc

        # ---- Gather decay and noise params per local gene index on device ---
        local_decay = xp.zeros(nGenes_layer, dtype=xp.float64)
        local_noise = xp.zeros(nGenes_layer, dtype=xp.float64)
        for li, gid in enumerate(geneIDs):
            local_decay[li] = self.decayVector_[gid]
            local_noise[li] = self.noiseParamsVector_[gid]

        # Reshape for broadcasting over bins: (nGenes_layer, 1)
        local_decay_2d = local_decay[:, None]
        local_noise_2d = local_noise[:, None]

        # Precompute constants
        sqrt_dt = xp.float64(np.sqrt(self.dt_))
        dt = xp.float64(self.dt_)

        # If there are target genes, we need a helper to look up regulator
        # concentrations.  Regulators may live in *this* layer or an earlier
        # (already finished) one.  We build a mapping: for each (target, reg_slot)
        # pair tell us whether the regulator lives in this layer (and its local
        # index) or not (use meanExpression directly).
        if nTargets > 0 and max_regs > 0:
            # For regulators in the *current* layer we read from conc_history.
            # For regulators in a *previous* layer we assume their mean expression
            # across bins is constant (already in meanExpression and used for
            # half-response).  But actually, during simulation the original code
            # reads the *current step concentration* of the regulator if it's in
            # the same layer.  We replicate that.
            #
            # Build lookup: reg_local_idx[ti, ri] = local index in this layer
            #               or -1 if the regulator is in a different layer.
            geneID_to_local = {gid: li for li, gid in enumerate(geneIDs)}
            reg_local_idx = -1 * np.ones((nTargets, max_regs), dtype=np.intp)
            reg_is_local = np.zeros((nTargets, max_regs), dtype=np.bool_)
            for ti, gid in enumerate(target_gene_ids):
                for ri, (rId, K, cs, hr) in enumerate(self.graph_[gid]['params']):
                    if rId in geneID_to_local:
                        reg_local_idx[ti, ri] = geneID_to_local[rId]
                        reg_is_local[ti, ri] = True

            reg_local_idx_d = xp.asarray(reg_local_idx)
            reg_is_local_d = xp.asarray(reg_is_local)

            # For non-local regulators, store their *current step conc* from
            # meanExpression, which doesn't change step-to-step.
            # Shape: (nTargets, max_regs, nBins) — only entries where
            # reg_is_local==False matter.
            nonlocal_reg_conc = xp.zeros((nTargets, max_regs, nBins), dtype=xp.float64)
            for ti, gid in enumerate(target_gene_ids):
                for ri, (rId, K, cs, hr) in enumerate(self.graph_[gid]['params']):
                    if rId not in geneID_to_local:
                        # regulator finished in a previous layer — use the
                        # *per-bin concentration history sampling*.
                        # The original code reads `regGene_allBins[bIdx].Conc[currStep]`
                        # which during simulation is just the latest conc.
                        # After a layer finishes, the mean expression is set.
                        # The original also has this *fixed* once the layer is
                        # done.  So we use meanExpression here.
                        nonlocal_reg_conc[ti, ri, :] = self.meanExpression[rId, :]

        # ---- Time-stepping loop -------------------------------------------
        print(f"  Simulating {nGenes_layer} genes × {nBins} bins for {nReqSteps} steps …")
        for step in range(1, nReqSteps):
            curr_conc = conc_history[step - 1]  # (nGenes_layer, nBins)

            # ---------- Production rate: (nGenes_layer, nBins) ---------------
            prod_rate = xp.zeros_like(curr_conc)

            # Master regulators: constant production rate
            for local_idx in mr_mask:
                gid = geneIDs[local_idx]
                prod_rate[local_idx, :] = xp.asarray(self.graph_[gid]['rates'], dtype=xp.float64)

            # Target genes: vectorised Hill + matmul
            if nTargets > 0 and max_regs > 0:
                # Gather regulator concentrations: (nTargets, max_regs, nBins)
                reg_conc = xp.zeros((nTargets, max_regs, nBins), dtype=xp.float64)

                # Local regulators – fancy-index into curr_conc
                # Build flat indices for local regulators
                for ti in range(nTargets):
                    nr = int(n_regs_per_gene[ti])
                    for ri in range(nr):
                        if reg_is_local[ti, ri]:
                            li = reg_local_idx[ti, ri]
                            reg_conc[ti, ri, :] = curr_conc[li, :]
                        else:
                            reg_conc[ti, ri, :] = nonlocal_reg_conc[ti, ri, :]

                # Vectorised Hill function: (nTargets, max_regs, nBins)
                hill_vals = self._hill_vectorised(
                    reg_conc, h_resp_d[:, :, None], coop_d[:, :, None],
                    is_repressive_d[:, :, None]
                )
                # Zero out padded slots
                hill_vals = hill_vals * reg_mask_d[:, :, None]

                # Weighted sum: for each target gene, sum K_abs[ti,ri] * hill[ti,ri,bin]
                # prod[ti, bin] = sum_ri  K[ti,ri] * hill[ti,ri,bin]
                # = einsum('tr,trb->tb', K_abs, hill_vals)
                target_prod = xp.einsum('tr,trb->tb', K_abs_d, hill_vals)

                # Write into prod_rate at the target positions
                for ti, local_idx in enumerate(target_mask):
                    prod_rate[local_idx, :] = target_prod[ti, :]

            # ---------- Decay: (nGenes_layer, nBins) -------------------------
            decay = local_decay_2d * curr_conc

            # ---------- Noise: (nGenes_layer, nBins) -------------------------
            if self.noiseType_ == 'sp':
                dw = xp.random.normal(size=(nGenes_layer, nBins))
                amplitude = local_noise_2d * xp.sqrt(xp.maximum(prod_rate, 0.0))
                noise = amplitude * dw

            elif self.noiseType_ == 'spd':
                dw = xp.random.normal(size=(nGenes_layer, nBins))
                amplitude = local_noise_2d * (
                    xp.sqrt(xp.maximum(prod_rate, 0.0)) +
                    xp.sqrt(xp.maximum(decay, 0.0))
                )
                noise = amplitude * dw

            elif self.noiseType_ == 'dpd':
                dw_p = xp.random.normal(size=(nGenes_layer, nBins))
                dw_d = xp.random.normal(size=(nGenes_layer, nBins))
                amp_p = local_noise_2d * xp.sqrt(xp.maximum(prod_rate, 0.0))
                amp_d = local_noise_2d * xp.sqrt(xp.maximum(decay, 0.0))
                noise = amp_p * dw_p + amp_d * dw_d
            else:
                noise = xp.zeros_like(curr_conc)

            # ---------- Euler-Maruyama step ----------------------------------
            new_conc = curr_conc + dt * (prod_rate - decay) + sqrt_dt * noise
            # Clamp to non-negative
            xp.maximum(new_conc, 0.0, out=new_conc)

            conc_history[step] = new_conc

        # ---- Sample single-cell expressions --------------------------------
        # scIndices_ contains negative indices into the *end* of the trajectory
        sc_indices = self.scIndices_  # NumPy array of negative ints
        # Convert to positive indices within conc_history
        pos_indices = nReqSteps + sc_indices  # still numpy

        for li, gid in enumerate(geneIDs):
            sc_expr = xp.zeros((nBins, self.nSC_), dtype=xp.float64)
            for si, pidx in enumerate(pos_indices):
                sc_expr[:, si] = conc_history[pidx, li, :]
            self.scExpressions_[gid] = sc_expr
            # Set meanExpression for downstream layers
            self.meanExpression[gid, :] = xp.mean(sc_expr, axis=1)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _hill_scalar(reg_conc, half_response, coop_state, repressive=False):
        """Scalar Hill function (used only for initialisation)."""
        if reg_conc == 0:
            return 1.0 if repressive else 0.0
        h = np.power(reg_conc, coop_state) / (
            np.power(half_response, coop_state) + np.power(reg_conc, coop_state)
        )
        return (1.0 - h) if repressive else h

    @staticmethod
    def _hill_vectorised(reg_conc, half_response, coop_state, repressive):
        """
        Batched Hill function over arrays.

        Parameters
        ----------
        reg_conc     : xp array, any shape
        half_response: xp array, broadcastable
        coop_state   : xp array, broadcastable
        repressive   : xp bool array, broadcastable

        Returns
        -------
        xp array of same shape as reg_conc
        """
        # Avoid divide-by-zero: where reg_conc==0 the Hill value is 0 (or 1 if repressive)
        safe_conc = xp.maximum(reg_conc, 1e-30)
        safe_hr = xp.maximum(half_response, 1e-30)
        numerator = xp.power(safe_conc, coop_state)
        denominator = xp.power(safe_hr, coop_state) + numerator
        h = numerator / denominator

        # Where reg_conc was truly 0, set h=0
        h = xp.where(reg_conc <= 0, xp.zeros_like(h), h)

        # Handle repressive interactions: 1 - h
        result = xp.where(repressive, 1.0 - h, h)
        return result

    # -------------------------------------------------------------- outputs

    def getExpressions(self):
        """Return expression array with shape (nBins, nGenes, nSC)."""
        ret = xp.zeros((self.nBins_, self.nGenes_, self.nSC_), dtype=xp.float64)
        for gid in range(self.nGenes_):
            if gid in self.scExpressions_:
                ret[:, gid, :] = self.scExpressions_[gid]
        return _to_numpy(ret)

    # -------------------------------------------------------------- technical noise

    def outlier_effect(self, scData, outlier_prob, mean, scale):
        out_indicator = np.random.binomial(n=1, p=outlier_prob, size=self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)

        scData = np.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx, :] = scData[gIndx, :] * outFactors[i]
        return np.split(scData, self.nBins_, axis=1)

    def lib_size_effect(self, scData, mean, scale):
        ret_data = []
        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.nBins_, self.nSC_))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis=0)
            ret_data.append(np.multiply(binExprMatrix, binFactors))
        return libFactors, np.array(ret_data)

    def dropout_indicator(self, scData, shape=1, percentile=65):
        scData = np.array(scData)
        scData_log = np.log(np.add(scData, 1))
        log_mid_point = np.percentile(scData_log, percentile)
        prob_ber = np.true_divide(1, 1 + np.exp(-1 * shape * (scData_log - log_mid_point)))
        binary_ind = np.random.binomial(n=1, p=prob_ber)
        return binary_ind

    def convert_to_UMIcounts(self, scData):
        return np.random.poisson(scData)
