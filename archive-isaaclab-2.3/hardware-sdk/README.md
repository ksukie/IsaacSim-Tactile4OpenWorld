# External OpenWorldTactile camera SDK

The historical experiments expect a separately obtained camera SDK at:

```text
archive-isaaclab-2.3/hardware-sdk/openworldtactile/
```

The previously bundled SDK was excluded from the public distribution on 2026-07-22 because it did not include a verifiable license grant and contained vendor-native `SonixCamera.dll` and `libSonixCamera.so` binaries. The old README's MIT badge pointed to a missing license file and was not treated as sufficient redistribution permission.

Obtain the SDK from its rightsholder or an authorized distributor, verify the version and platform, and place it at the path above or set `OWT_SDK_ROOT` where supported. Do not commit it unless written redistribution rights, license text, source/version provenance, and binary notices have been documented.

The SDK-dependent legacy scripts are preserved for reproducibility but are not part of the default open-source runtime path.
