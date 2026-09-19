# Oriđiđi web asset provenance

The unserved source is `src/oridjidji/web/ori-logo.gif`, a 1200×1200,
150-frame, five-second animation with SHA-256
`195884aafd610f573c948399b8130af1f52b4d701cdf072f11147a4512b43c6d`.
It remains on the author's machine and is gitignored because it is 27.4 MB.

| Served file | Derivation | Size | SHA-256 |
|---|---|---:|---|
| `oridjidji-logo-animated.webp` | every second source frame; 240×240; 75 frames; five seconds; one loop | 710,276 B | `981b416511c69f0da70c90b377d8c0e8b4c19b620f7d8ec4621a0a4bbb3fd5a4` |
| `oridjidji-logo-static.webp` | final source frame; 480×480; lossless WebP | 201,400 B | `ceb2cd6d9e498f931d3b44201e0d450b489125c8caf8e239e87cef78401fd1e9` |
| `oridjidji-mark.svg` | compact ring-and-cursor mark using the source geometry and brand palette | 701 B | `7b26a3056def338d65a6b0ff6782116db8e4ee80fbebec6b7d9331ade02ae272` |
| `oridjidji-logo-v2.webp` | `web/logo.png`; 700×520; lossless WebP | 241,626 B | `86276a0cabd652e01a30b7cbfa356f460804f3d457c549ba27cd04f42a2cb262` |
| `oridjidji-wordmark.webp` | `web/text logo.png`; 512×128; lossless WebP | 9,374 B | `7588d9696a100c204c2466fb231ab49c3437b3f987b5d9d0652e3c646e7bec1b` |
| `hero-halftone.webp` | `web/elelemnt.png`; 1000×1000; WebP quality 86 | 237,250 B | `3b184fc810dc0060e1db67555d72322e53c8abb7ac0d51225005db4a2391cad2` |
| `element-animated.webp` | every third frame of `web/element.gif`; 640×640; 50 frames; ten-second loop | 5,751,564 B | `fb4a221736c887ea54d20150d2ca02e5add6b9ce05a05774ff17907b174b2b85` |
| `element-static.webp` | final selected frame of `web/element.gif`; 640×640 | 163,114 B | `a8f45e547a29b62a2bccfbe081fefadfbd24e203be7c3d5e99149c65b7fba53a` |

The header shows `oridjidji-logo-v2.webp` at the top of the page and flips to
`oridjidji-wordmark.webp` after scrolling. The older logo animation and
its fallback remain preserved as historical design assets. The analysis-page
introduction serves `element-static.webp` and rotates that single image with CSS;
the frame animation is retained but unserved. The 69.6 MB source GIF remains unserved and has
SHA-256 `a26a0079cb77e4bb2bdb9f45283d4ec2dc72645131fb68fc06cb5377aaf001d5`.
