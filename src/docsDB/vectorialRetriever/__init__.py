from langchain_classic.chains.query_constructor.base import AttributeInfo

METADATA_FIELD_INFO = [AttributeInfo(name="destination",
                                     description="The specific UK destination to be searched",
                                     type="string"),
                       AttributeInfo(name="region",
                                     description="The name of the UK region to be searched",
                                     type="string")]

DEF_K = 2