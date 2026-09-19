# Offline font provenance

Downloaded on 2026-09-13 from the browser-targeted CSS returned by the official
Google Fonts endpoint. The files are served only by the localhost application.

| Local file | Official source | SHA-256 |
|---|---|---|
| `archivo-latin-ext.woff2` | `fonts.gstatic.com/s/archivo/v25/k3kPo8UDI-1M0wlSV9XAw6lQkqWY8Q82sLyTOxK-vA.woff2` | `ff4f17d21930e36d6d93baba663e624cb767afc3feebf7adaebd82242638de05` |
| `archivo-latin.woff2` | `fonts.gstatic.com/s/archivo/v25/k3kPo8UDI-1M0wlSV9XAw6lQkqWY8Q82sLydOxI.woff2` | `8f704806dbedeaaeca334b11ec348bc3ac3a439d6431544b3afb54f534ee4967` |
| `ibm-plex-mono-400-latin-ext.woff2` | `fonts.gstatic.com/s/ibmplexmono/v20/-F63fjptAgt5VM-kVkqdyU8n1iEq129k.woff2` | `6bc0f226a5b7884a8170e3f62c63d7675609d4631bdc5931b5cdab81821f00eb` |
| `ibm-plex-mono-400-latin.woff2` | `fonts.gstatic.com/s/ibmplexmono/v20/-F63fjptAgt5VM-kVkqdyU8n1i8q1w.woff2` | `08949f728dc52d528e69b1667d15c89a5686a4ee9a296ff90983985f99c380f7` |
| `ibm-plex-mono-500-latin-ext.woff2` | `fonts.gstatic.com/s/ibmplexmono/v20/-F6qfjptAgt5VM-kVkqdyU8n3twJwl5FgtIU.woff2` | `6bb06407c97584b0867a959e05e8874693bfeb8c317de190811c51598f2d99ea` |
| `ibm-plex-mono-500-latin.woff2` | `fonts.gstatic.com/s/ibmplexmono/v20/-F6qfjptAgt5VM-kVkqdyU8n3twJwlBFgg.woff2` | `01d285447409c8a588692162439a038b8cbd7871309ee20267b0d2d91c6e8e22` |
| `ibm-plex-sans-latin-ext.woff2` | `fonts.gstatic.com/s/ibmplexsans/v23/zYXzKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1syxQKYbABA.woff2` | `d160e20920ae4d6556518d352d3af27a74e9b0de3d8fe17b1c1044fc75aa2f81` |
| `ibm-plex-sans-latin.woff2` | `fonts.gstatic.com/s/ibmplexsans/v23/zYXzKVElMYYaJe8bpLHnCwDKr932-G7dytD-Dmu1syxeKYY.woff2` | `e2291e842cf5af167122a22881a740c7f2dda7716f1e8cd76680264f4a859470` |

The SIL Open Font License 1.1 texts were downloaded from the matching Archivo,
IBM Plex Sans and IBM Plex Mono directories in the official `google/fonts`
repository and are stored beside these files.

The full official font files were checked with `fontTools` for
`č ć đ š ž Č Ć Đ Š Ž`; every requested face and weight contained all ten glyphs.
