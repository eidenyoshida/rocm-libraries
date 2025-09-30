/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            MemoryType GetMemoryType(LoadMode const& mode)
            {
                switch(mode)
                {
                case LoadMode::VGPR:
                    return MemoryType::WAVE;
                case LoadMode::VGPRToLDS:
                    return MemoryType::WAVE_LDS;
                case LoadMode::BufferToLDS:
                    return MemoryType::WAVE_Direct2LDS;
                case LoadMode::Count:
                    Throw<FatalError>(fmt::format("No valid MemoryType available for LDS mode {}\n",
                                                  toString(mode)));
                }
            }

            bool IsBufferToLDS(LoadMode const& mode)
            {
                return mode == LoadMode::BufferToLDS;
            }

            std::string toString(LoadMode mode)
            {
                switch(mode)
                {
                case LoadMode::VGPR:
                    return "VGPR";
                case LoadMode::VGPRToLDS:
                    return "VGPRToLDS";
                case LoadMode::BufferToLDS:
                    return "BufferToLDS";
                default:
                    break;
                }
                return "Invalid";
            }

            std::ostream& operator<<(std::ostream& stream, LoadMode const& mode)
            {
                return stream << toString(mode);
            }

            std::istream& operator>>(std::istream& stream, LoadMode& mode)
            {
                std::string strValue;
                stream >> strValue;

                if(strValue == toString(LoadMode::VGPR))
                {
                    mode = LoadMode::VGPR;
                }
                else if(strValue == toString(LoadMode::VGPRToLDS))
                {
                    mode = LoadMode::VGPRToLDS;
                }
                else if(strValue == toString(LoadMode::BufferToLDS))
                {
                    mode = LoadMode::BufferToLDS;
                }
                else
                {
                    Throw<FatalError>(fmt::format("Invalid LoadMode {}\n", strValue));
                }

                return stream;
            }
        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
