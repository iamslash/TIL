
- [Block](#block)
- [Generate Block](#generate-block)
- [Proof of work](#proof-of-work)

----

# Block

```cpp
```

# Generate Block

```cpp
// src/rpc/mining.cpp

static bool GenerateBlock(ChainstateManager& chainman, CBlock& block, uint64_t& max_tries, unsigned int& extra_nonce, uint256& block_hash)
{
    block_hash.SetNull();

    {
        LOCK(cs_main);
        IncrementExtraNonce(&block, chainman.ActiveChain().Tip(), extra_nonce);
    }

    CChainParams chainparams(Params());

    while (max_tries > 0 && block.nNonce < std::numeric_limits<uint32_t>::max() && !CheckProofOfWork(block.GetHash(), block.nBits, chainparams.GetConsensus()) && !ShutdownRequested()) {
        ++block.nNonce;
        --max_tries;
    }
    if (max_tries == 0 || ShutdownRequested()) {
        return false;
    }
    if (block.nNonce == std::numeric_limits<uint32_t>::max()) {
        return true;
    }

    std::shared_ptr<const CBlock> shared_pblock = std::make_shared<const CBlock>(block);
    if (!chainman.ProcessNewBlock(chainparams, shared_pblock, true, nullptr)) {
        throw JSONRPCError(RPC_INTERNAL_ERROR, "ProcessNewBlock, block not accepted");
    }

    block_hash = block.GetHash();
    return true;
}
```

# Proof of work

```cpp
// src/pow.cpp

bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params& params)
{
    bool fNegative;
    bool fOverflow;
    arith_uint256 bnTarget;

    bnTarget.SetCompact(nBits, &fNegative, &fOverflow);

    // Check range
    if (fNegative || bnTarget == 0 || fOverflow || bnTarget > UintToArith256(params.powLimit))
        return false;

    // Check proof of work matches claimed amount
    if (UintToArith256(hash) > bnTarget)
        return false;

    return true;
}
```
