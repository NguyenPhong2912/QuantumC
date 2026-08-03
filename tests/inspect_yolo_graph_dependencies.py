from ultralytics import YOLO


def normalize_sources(source):
    if isinstance(source, int):
        return [source]
    return list(source)


def main() -> None:
    print("===== YOLO GRAPH DEPENDENCY CHECK =====")

    model = YOLO("yolo11n.yaml")
    layers = model.model.model

    target_layer = 10
    consumers = []

    for index, layer in enumerate(layers):
        sources = normalize_sources(getattr(layer, "f", -1))

        resolved_sources = []

        for source in sources:
            if source == -1:
                resolved = index - 1
            else:
                resolved = source

            resolved_sources.append(resolved)

            if resolved == target_layer:
                consumers.append(index)

        print(
            f"Layer {index:>2} "
            f"{layer.__class__.__name__:<20} "
            f"from={getattr(layer, 'f', None)!s:<14} "
            f"resolved={resolved_sources}"
        )

    print()
    print("Target layer         :", target_layer)
    print("Target module        :", layers[target_layer].__class__.__name__)
    print("Direct consumers     :", consumers)

    assert layers[target_layer].__class__.__name__ == "C2PSA"
    assert 11 in consumers
    assert 21 in consumers

    print("Selected placement   : after Layer 10")
    print("Expected channels    : 256")
    print("Expected resolution  : 20 x 20 at input 640")
    print("GRAPH DEPENDENCY TEST: PASSED")


if __name__ == "__main__":
    main()
