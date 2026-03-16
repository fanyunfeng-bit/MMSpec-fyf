import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)


class MSDDynamicCache:
    """
    A wrapper around HuggingFace's DynamicCache that provides MSD-compatible interface.
    
    This class allows MSD to work with HuggingFace transformers models that expect
    DynamicCache objects for past_key_values.
    """
    
    def __init__(self, num_layers: int, device: torch.device, dtype: torch.dtype):
        """
        Initialize MSDDynamicCache.
        
        Args:
            num_layers: Number of transformer layers
            device: Device for tensors
            dtype: Data type for tensors
        """
        from transformers import DynamicCache
        
        self.cache = DynamicCache()
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self._current_length = 0
    
    def get_hf_cache(self):
        """Return the underlying HF DynamicCache for model forward pass."""
        return self.cache
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length in the cache."""
        return self.cache.get_seq_length(layer_idx)
    
    def reset(self):
        """Reset the cache to empty state."""
        from transformers import DynamicCache
        self.cache = DynamicCache()
        self._current_length = 0
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        """
        Update cache for a specific layer.
        
        Args:
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            
        Returns:
            Tuple of updated (key_states, value_states)
        """
        return self.cache.update(key_states, value_states, layer_idx)
    
    def get_key_value(self, layer_idx: int):
        """Get key-value pair for a specific layer."""
        if layer_idx < len(self.cache.key_cache):
            return self.cache.key_cache[layer_idx], self.cache.value_cache[layer_idx]
        return None, None
    
    def truncate(self, max_length: int):
        """Truncate cache to max_length."""
        for layer_idx in range(len(self.cache.key_cache)):
            if self.cache.key_cache[layer_idx] is not None:
                self.cache.key_cache[layer_idx] = self.cache.key_cache[layer_idx][:, :, :max_length, :]
                self.cache.value_cache[layer_idx] = self.cache.value_cache[layer_idx][:, :, :max_length, :]
        self._current_length = max_length
    
    def select_indices(self, indices: torch.Tensor, prev_length: int):
        """
        Select specific indices from cache and place at new positions.
        Used for tree attention verification.
        
        Args:
            indices: Token indices to select
            prev_length: Previous sequence length
        """
        for layer_idx in range(len(self.cache.key_cache)):
            if self.cache.key_cache[layer_idx] is not None:
                key = self.cache.key_cache[layer_idx]
                value = self.cache.value_cache[layer_idx]
                
                # Select keys/values at specified indices
                selected_keys = key[:, :, indices, :]
                selected_values = value[:, :, indices, :]
                
                # Create new cache with selected tokens
                new_key = key[:, :, :prev_length, :].clone()
                new_value = value[:, :, :prev_length, :].clone()
                
                # Append selected tokens
                new_key = torch.cat([new_key, selected_keys], dim=2)
                new_value = torch.cat([new_value, selected_values], dim=2)
                
                self.cache.key_cache[layer_idx] = new_key
                self.cache.value_cache[layer_idx] = new_value
        
        self._current_length = prev_length + len(indices)


def initialize_past_key_values_hf(model, use_dynamic=True):
    """
    Initialize past key values for HuggingFace models using DynamicCache.
    
    Args:
        model: The HF model (LlavaForConditionalGeneration)
        use_dynamic: Whether to use DynamicCache (True) or None for first pass
        
    Returns:
        DynamicCache object or None
    """
    if use_dynamic:
        from transformers import DynamicCache
        return DynamicCache()
    return None


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    # Handle HF LLaVA which has nested text_config
    config = model.config
    if hasattr(config, 'text_config'):
        # HF LLaVA: use text_config for language model parameters
        lm_config = config.text_config
    else:
        lm_config = config

    if not hasattr(lm_config, 'num_key_value_heads'):   # QWen or older models
        lm_config.num_key_value_heads = lm_config.num_attention_heads
        lm_config.max_position_embeddings = lm_config.max_position_embeddings // 2

    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers

    # Get language model for device detection
    if hasattr(model, 'language_model'):
        lm_model = model.language_model
    else:
        lm_model = model
    
    devices=[]
    for i in range(lm_config.num_hidden_layers):
        if hasattr(lm_model, "model") and hasattr(lm_model.model, "layers"):
            device = lm_model.model.layers[i].self_attn.q_proj.weight.device
        elif hasattr(lm_model, "model") and hasattr(lm_model.model, "h"):
            device = lm_model.model.h[-1].attn.c_attn.weight.device
        elif hasattr(lm_model, "layers"):
            device = lm_model.layers[i].self_attn.q_proj.weight.device
        elif hasattr(lm_model, "h"):
            device = lm_model.h[-1].attn.c_attn.weight.device
        devices.append(device)

    past_key_values_data_list=[]
    startnum=0
    startdevice=devices[0]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                lm_config.num_key_value_heads,
                lm_config.max_position_embeddings,
                lm_config.hidden_size // lm_config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        lm_config.num_key_value_heads,
        lm_config.max_position_embeddings,
        lm_config.hidden_size // lm_config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        lm_config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * lm_config.num_hidden_layers

    bias=0
    start_data_m=devices[0].index
    for i in range(lm_config.num_hidden_layers):

        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1
    return past_key_values, past_key_values_data_list, current_length_data
