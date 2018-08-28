import utils
from . import common
from . import manager
from . import nonlinear

MODES = ["word", "label", "intent"]


def add_arguments(parser):
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--default-nonlinear", type=str, default="tanh",
                        choices=manager.get_module_names(nonlinear))
    parser.add_argument("--z-dim", type=int, default=300)
    for mode in MODES:
        parser.add_argument(f"--{mode}-dim", type=int, default=300)
        parser.add_argument(f"--{mode}-freeze", action="store_true",
                            default=False)


def create_model(args, vocabs):
    namemaps = manager.get_module_namemap(nonlinear)
    nonlinear_cls = namemaps[args.default_nonlinear]
    nonlinear.set_default(nonlinear_cls)
    return create_luvae(args, vocabs)


def create_mmvae(args, vocabs):
    mmvae = utils.import_module(f"model.mmvae")
    model_cls = mmvae.MultimodalVariationalAutoencoder
    if args.model_path is None:
        modargs = get_optarg_template(model_cls)
        modargs["encoders"] *= 3
        modargs["decoders"] *= 3
    else:
        modargs = utils.load_yaml(args.model_path)
    for enc, dec, vocab, mode in \
            zip(modargs["encoders"], modargs["decoders"], vocabs, MODES):
        freeze = getattr(args, f"{mode}_freeze", False)
        unfrozen_idx = {vocab[w] for w in [args.bos, args.eos, args.unk]}
        for coder in (enc, dec):
            coder["vargs"]["embed"] = {
                "type": "finetunable-embedding",
                "vargs": {
                    "unfrozen_idx": unfrozen_idx,
                    "freeze": freeze,
                    "allow_padding": True
                }
            }
    dim_keys = [f"{mode}_dim" for mode in MODES]
    dims = [getattr(args, k, 300) for k in dim_keys]
    z_dim = getattr(args, "z_dim", 300)
    caster = common.get_caster(model_cls)
    creator = caster({
        "type": "multimodal-variational-autoencoder",
        "vargs": modargs
    })
    return creator(
        num_modes=len(MODES),
        z_dim=z_dim,
        vocab_sizes=[len(vocab) for vocab in vocabs],
        word_dims=dims
    )


def create_luvae(args, vocabs):
    luvae = utils.import_module(f"model.luvae")
    if args.model_path is None:
        model_cls = manager.get_module_classes(luvae)[0]
        modargs = get_optarg_template(model_cls)
    else:
        namemap = manager.get_module_namemap(luvae)
        opts = utils.load_yaml(args.model_path)
        name, modargs = opts.get("type"), opts.get("vargs")
        model_cls = namemap[name]
    def setup_embed(vocab, freeze):
        unfrozen_idx = {vocab[w] for w in [args.bos, args.eos, args.unk]}
        return {
            "type": "finetunable-embedding",
            "vargs": {
                "unfrozen_idx": unfrozen_idx,
                "freeze": freeze,
                "allow_padding": True
            }
        }
    for vocab, mode in zip(vocabs, MODES):
        modargs[f"{mode}_embed"] = \
            setup_embed(vocab, getattr(args, f"{mode}_freeze"))
    dim_keys = [f"{mode}_dim" for mode in MODES]
    dims = [getattr(args, k, 300) for k in dim_keys]
    z_dim = getattr(args, "z_dim", 300)
    caster = common.get_caster(model_cls)
    creator = caster({
        "type": model_cls.name,
        "vargs": modargs
    })
    return creator(
        z_dim=z_dim,
        word_dim=dims[0],
        label_dim=dims[1],
        intent_dim=dims[2],
        num_words=len(vocabs[0]),
        num_labels=len(vocabs[1]),
        num_intents=len(vocabs[2])
    )


def get_optarg_template(cls: common.Module):
    def get_value_template(optarg: common.OptionalArgument):
        if optarg.islist:
            sample = optarg.default[0]
        else:
            sample = optarg.default
        if common.is_module_cls(sample):
            pkg = sample.get_package()
            cls = manager.get_module_classes(pkg)[0]
            val = {"type": cls.name}
            args = get_optarg_template(cls)
            if args:
                val["vargs"] = args
        else:
            val = sample
        if optarg.islist:
            val = [val]
        return val
    return {
        name: get_value_template(optarg)
        for name, optarg in cls.get_optargs().items()
    }