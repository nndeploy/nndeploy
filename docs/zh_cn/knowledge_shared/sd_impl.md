
# stable diffusion impl

- 1. Tokenizer
- 2. CLIP
- 3. UNET
  - 3.1 策略
  - 3.2 模型结构
- 4. VAE Decodec
- 5. lora
- 6. control net
- 7. 应用场景
- 8. 提示词

## Tokenizer

采用的是CLIPTokenizer。以下是对CLIPTokenizer的描述：
CLIP（Contrastive Language-Image Pre-training）是一个多模态学习框架，它通过联合学习图像和文本来捕捉视觉内容和语言之间的联系。CLIP模型由OpenAI提出，旨在使机器能够理解图像和文本之间的关系。CLIPTokenizer是用于处理文本数据，将其转换为模型可以处理的格式的工具。

CLIP模型的文本部分通常基于Transformer架构，因此CLIPTokenizer遵循了大多数基于Transformer的模型（如BERT）的文本处理流程。以下是CLIPTokenizer的详解：

1. **分词（Tokenization）**：CLIPTokenizer将输入的文本字符串分割成更小的单元，称为tokens。这通常涉及到将单词分割成子词（subwords）或标记（tokens），以便模型可以更有效地处理。

2. **特殊标记（Special Tokens）**：为了表示序列的开始和结束，CLIPTokenizer会添加特殊的标记，如`[CLS]`和`[SEP]`。这些标记对于模型理解序列结构至关重要。

3. **填充（Padding）**：由于模型通常需要固定长度的输入，CLIPTokenizer会将较短的序列填充到与序列中最长文本相同的长度。

4. **序列化（Truncation）**：如果文本长度超过模型允许的最大长度，CLIPTokenizer会截断序列，以确保它符合模型的输入要求。

5. **建立注意力掩码（Attention Mask）**：CLIPTokenizer生成一个掩码来指示模型应该关注序列中的哪些部分。这通常用于区分实际的文本tokens和填充tokens。

6. **建立类型掩码（Type Mask）**：在多模态学习中，CLIPTokenizer可能会生成一个类型掩码来区分不同模态的输入，例如区分文本和图像特征。

7. **转换为ID（Token to ID conversion）**：CLIPTokenizer将tokens转换为模型可以理解的整数ID，这些ID对应于模型词汇表中的特定词汇。

8. **输出**：最终，CLIPTokenizer输出一个包含tokens ID、注意力掩码、类型掩码和可能的其他信息的字典，这些信息可以直接用于模型的输入。

CLIPTokenizer的具体实现细节可能会根据所使用的模型版本和库（如Hugging Face的Transformers库）有所不同。在实际应用中，开发者通常会使用现有的库来处理这些文本预处理任务，因为它们提供了高效的实现和对不同模型的广泛支持。

如果你需要一个具体的代码示例或者想了解如何使用某个特定的库（如Hugging Face的Transformers）来实现CLIPTokenizer，请提供更多的上下文，我会根据你的需要提供相应的信息。