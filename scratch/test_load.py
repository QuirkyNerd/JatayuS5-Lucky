import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

import asyncio
from services.guideline_loader import GuidelineLoader

async def main():
    loader = GuidelineLoader()
    print("data_dir:", loader.data_dir)
    print("cpt_codes.csv exists:", (loader.data_dir / "cpt_codes.csv").exists())
    print("Reingesting CPT...")
    count = await loader.reingest_cpt()
    print("CPT loaded count:", count)

if __name__ == "__main__":
    asyncio.run(main())
