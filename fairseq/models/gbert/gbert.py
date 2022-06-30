from fairseq.models import BaseFairseqModel, register_model
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from fairseq.models import register_model_architecture

@register_model('gbert')
class Gbert(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.gbert = encoder

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add a new command-line argument to configure the
        # dimensionality of the hidden state.
        parser.add_argument(
            '--in_features', type=int, metavar='N',
            help='dimensionality of the in_features',
        )
        parser.add_argument(
            '--model_name', type=str, required=True,
            help='name of the model',
        )

        parser.add_argument(
            '--pretrained', type=bool, default=True,
            help='name of the model',
        )
        parser.add_argument(
            '--num_classes', type=int, default=1000,
            help='name of the model',
        )
        parser.add_argument(
            '--drop', type=float, default=0.1,
            help='drop',
        )
        parser.add_argument(
            '--drop_path', type=float, default=0.1,
            help='drop_path',
        )
        parser.add_argument(
            '--drop_path_rate', type=float, default=0.1,
            help='drop_block',
        )
        parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                            help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

        # Dataset / Model parameters
        parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                            help='Initialize model from this checkpoint (default: none)')

        # Batch norm parameters (only works with gen_efficientnet based models currently)
        parser.add_argument('--bn-tf', action='store_true', default=False,
                            help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
        parser.add_argument('--bn-momentum', type=float, default=None,
                            help='BatchNorm momentum override (if not None)')
        parser.add_argument('--bn-eps', type=float, default=None,
                            help='BatchNorm epsilon override (if not None)')

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.
        gbert = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path=args.drop_path,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)
        return cls(args, gbert)

    def forward(self, inputs):
        return self.gbert(inputs)

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.
@register_model_architecture('gbert', 'gbert')
def base_architecture(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.hidden_dim = getattr(args, 'hidden_dim', 128)